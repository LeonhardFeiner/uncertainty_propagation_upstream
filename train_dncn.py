import torch
import merlinpy
import os
import sys
from pathlib import Path
from numpy import ceil
import fastmri_dataloader
from fastmri_dataloader.fastmri_dataloader_th import FastmriCartesianDataset
from fastmri.models.dncn import DnCn

from torch.utils.data import DataLoader, dataloader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
import wandb
import logging
from tqdm import tqdm

# def collate_fn_crop(batch):
#     '''
#     crop batch of variable size
#     '''
    
#     ## get sequence lengths
#     lengths = torch.tensor([ t.shape[0] for t in batch ]).to(device)
#     ## padd
#     batch = [ torch.Tensor(t).to(device) for t in batch ]
#     batch = torch.nn.utils.rnn.pad_sequence(batch)
#     ## compute mask
#     mask = (batch != 0).to(device)
#     return batch, lengths, mask

REAL_BATCH_SIZE = 8

def get_args():
    parser = ArgumentParser()

    # training related
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--amp", default=False)
    parser.add_argument("--save_checkpoint", default=False)
    parser.add_argument("--load", default=None)


    # dataset
    #parser.add_argument("--challenge", default="singlecoil")
    #parser.add_argument("--data_path", default="/home/wenqi/Data/FastMRI/knee/data")
    #parser.add_argument("--mask_type", default="random")
    #parser.add_argument("--center_fractions", default=[0.08], type=float)
    #parser.add_argument("--accelerations", default=[4], type=int)
    #parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=str)

    return parser.parse_args()


def train(net, device, args):
    #os.environ['FASTMRI_ROOT'] = '/home/wenqi/Data/FastMRI'

    config_path = './fastmri_dataloader/config.yml'
    config = merlinpy.loadYaml(config_path, 'BaseExperiment')
    train_dataset = FastmriCartesianDataset(config, mode='singlecoil_train')
    val_dataset = FastmriCartesianDataset(config, mode='singlecoil_val')
    loader_args = dict(batch_size=1, num_workers=args.num_workers, pin_memory=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_dataloader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    checkpoint_path = './ckpt'

    # optimizer related
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.L1Loss()

    global_step = 0

    n_train = len(train_dataset) * REAL_BATCH_SIZE
    n_val = len(val_dataset) * REAL_BATCH_SIZE

    experiment = wandb.init(project="dccnn-dp", reinit=False)
    experiment.config.update(dict(epochs=args.epochs,
                                  lr=args.lr, amp=args.amp))
    experiment.define_metric("train/*", step_metric="train/step")
    experiment.define_metric("val/*", step_metric="val/step")

    logging.info(
        f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {config['batch_size']}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {args.save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {args.amp}
    ''')

    #TODO logging info num parameters
    #TODO check loss?
    #TODO compute averaged epoch loss
    #TODO track psnr / ssim
    #TODO add UNET
    #TODO add param constraints, especially for lambda!

    for epoch in range(args.epochs):
        net.train()
        epoch_loss = 0
        
        with tqdm(total=n_train, desc=f'Train Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in train_dataloader:
                inputs, outputs = fastmri_dataloader.prepare_batch(batch, device)

                x0 = inputs[0]
                gnd = outputs[0]

                with torch.cuda.amp.autocast(enabled=args.amp):
                    output = net(*inputs)
                    loss = criterion(output.abs(), gnd.abs())
                
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)

                global_step += 1
                epoch_loss += loss.item()

                log_input_im = torch.flipud(x0[0].abs())
                log_output_im = torch.flipud(output[0].abs())
                log_gnd_im = torch.flipud(gnd[0].abs())
                log_im = torch.cat([log_input_im, log_output_im, log_gnd_im], dim=1) / log_gnd_im.max()
                log_im = torch.clamp_max(log_im, 1)
                #log_im = log_gnd_im / log_gnd_im.max()
                
                if global_step % 200 == 0:
                    experiment.log({
                        'train/loss': loss.item(),
                        'train/step': global_step,
                        'train/epoch': epoch,
                        'train/in_out_gnd': wandb.Image(log_im.float().cpu()),
                        'train/lr': optimizer.param_groups[0]['lr']
                    }, step=global_step)
                else:
                    experiment.log({
                        'train/loss': loss.item(),
                        'train/step': global_step,
                        'train/epoch': epoch,
                        'train/lr': optimizer.param_groups[0]['lr']
                    }, step=global_step)

                pbar.update(x0.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            

        # evaluation
        histograms = {}
        for tag, value in net.named_parameters():
            tag = tag.replace('/', '.')
            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            if value.requires_grad:
                histograms['Gradients' + tag] = wandb.Histogram(value.grad.data.cpu())
        
        

        net.eval()
        val_score = 0
        with torch.no_grad():
            with tqdm(total=n_val, desc=f'Val Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar_val:
                for idx, batch in enumerate(val_dataloader):
                    inputs, outputs = fastmri_dataloader.prepare_batch(batch, device)

                    x0 = inputs[0]
                    gnd = outputs[0]

                    output = net(*inputs)
                    val_loss = F.l1_loss(output.abs(), gnd.abs())

                    # logging.info('Validation {}/{} loss: {}'.format(idx, ceil(n_val / args.batch_size), val_loss))

                    log_input_im = torch.flipud(x0[0].abs())
                    log_output_im = torch.flipud(output[0].abs())
                    log_gnd_im = torch.flipud(gnd[0].abs())
                    log_im = torch.cat([log_input_im, log_output_im, log_gnd_im], dim=1) / log_gnd_im.max()
                    log_im = torch.clamp_max(log_im, 1)

                    experiment.log({
                        #'val/lr': optimizer.param_groups[0]['lr'],
                        'val/loss': val_loss,
                        'val/in_out_gnd': wandb.Image(log_im.float().cpu()),
                        'val/step': idx,
                        #'val/epoch': epoch,
                        **histograms
                    }, commit=False)
                    val_score += val_loss
                    pbar_val.update(x0.shape[0])
                    pbar_val.set_postfix(**{'loss (batch)': val_loss.item()})

                scheduler.step(val_score)



        if args.save_checkpoint:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(checkpoint_path / 'checkpoint_epoch{}.pth'.format(epoch+1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = DnCn()
    # net = DnCnComplexDP()
    logging.info(f'Network initialized!')
    
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device)
    try:
        train(net, device, args)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info(f'Saved interrupt')
        sys.exit(0)

    
