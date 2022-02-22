import torch
import os
import sys
from pathlib import Path
from numpy import ceil
import fastmri.data
from fastmri.data.fastmri_dataloader_th import FastmriCartesianDataset
from fastmri.models.dncn import DnCn
from fastmri.models.unet import Unet
from fastmri.models.snet import Snet
from fastmri.utils import loadYaml
import fastmri.losses
import fastmri.functional as F_fastmri

from torch.utils.data import DataLoader, dataloader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
import wandb
import logging
from tqdm import tqdm
from datetime import datetime

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


def get_args():
    parser = ArgumentParser()

    # training related
    parser.add_argument("--model", default='dncn', type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--amp", default=False)
    parser.add_argument("--save_checkpoint", default=False, type=bool)
    parser.add_argument("--load", default=None)


    # dataset
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--regularizer", default='Real2chCNN', type=str)
    parser.add_argument("--shared-params", action='store_true', help='Sharing paramters over cascades')
    parser.add_argument("--nc", type=int, default=10, help='Number of cascades')
    parser.add_argument("--nf", type=int, default=64, help='Num base features')
    parser.add_argument("--dropout", type=float, default=0, help="Dropout probability applied after most conv layers")
    parser.add_argument(
        "--epistemic",  action='store_true',
        help="Indicator whether network should model epistemic uncertainty. Needs dropout")
    parser.add_argument(
        "--num-samples", default=8, type=int,
        help="number of samples should be drawn fore calculating epistemic uncertainty. Needs dropout")
    parser.add_argument(
        "--aleatoric",  action='store_true', help="Indicator whether network should model aleatoric uncertainty")
    parser.add_argument(
        "--combined-aleatoric",  action='store_true',
        help="Indicator whether aleatoric prediction should use feature maps of every layer as input. Needs aleatoric")
    parser.add_argument(
        "--l2",  action='store_true',
        help="Indicator whether to use l2 loss instead of l1 loss")

    return parser.parse_args()


def train(net, device, args):
    config_path = './config.yml'
    config = loadYaml(config_path, 'BaseExperiment')
    train_dataset = FastmriCartesianDataset(config, mode='singlecoil_train')
    val_dataset = FastmriCartesianDataset(config, mode='singlecoil_val')
    loader_args = dict(batch_size=1, num_workers=args.num_workers, pin_memory=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_dataloader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    

    # optimizer related
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    if args.l2:
        if args.aleatoric:
            criterion = fastmri.losses.AttenuatedMaskedL2Loss()
        else:
            criterion = fastmri.losses.MaskedL2Loss()
    else:
        if args.aleatoric:
            criterion = fastmri.losses.AttenuatedMaskedL1Loss()
        else:
            criterion = fastmri.losses.MaskedL1Loss()
        

    global_step = 0

    n_train = len(train_dataset) * config['batch_size']
    n_val = len(val_dataset) * config['batch_size']

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    exp_id = '_'.join([TIMESTAMP, args.model])
    expconfig = vars(args)
    expconfig["total_params"] = total_params
    experiment = wandb.init(project="dccnn-dp", reinit=False,name=exp_id, config=expconfig)
    experiment.config.update(dict(epochs=args.epochs,
                                  lr=args.lr, amp=args.amp))
    checkpoint_path = Path('./ckpt') / exp_id
    #experiment.define_metric("train/*", step_metric="train/batch")
    #experiment.define_metric("val/*", step_metric="val/step")
    #experiment.define_metric('val_images/*', step_metric='val_images/idx')
    
    logging.info(
        f'''Starting training:
        Num_Params:      {total_params}
        Epochs:          {args.epochs}
        Batch size:      {config['batch_size']}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {args.save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {args.amp}
    ''')


    #TODO write RSS function (multicoil)
    #TODO didn needs lower learning rate trying with 2e-4
    #TODO test code
    #TODO correct training and validation steps in wandb

    for epoch in range(args.epochs):
        net.train()
        epoch_loss = 0
        
        with tqdm(total=n_train, desc=f'Train Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in train_dataloader:
                inputs, outputs = fastmri.data.prepare_batch(batch, device)
                
                x0 = inputs[0]
                gnd = outputs[0]
                fg_mask = inputs[-1] if config['use_fg_mask'] else torch.ones_like(x0, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=args.amp):
                    if config['use_fg_mask']:
                        selected_input = inputs[:-1] # pass all inputs except fg_mask
                    else:
                        selected_input = inputs

                    output = net(*selected_input)

                    if args.aleatoric:
                        output, raw_uncertainty = output
                        loss = criterion(output.abs(), gnd.abs(), fg_mask, raw_uncertainty)

                        if args.l2:
                            aleatoric_std = torch.sqrt(torch.exp(raw_uncertainty))
                        else:
                            aleatoric_std = torch.exp(raw_uncertainty) * torch.sqrt(raw_uncertainty.new_tensor(2))
                            # sigma ** 2 = 2 * b ** 2

                    else:
                        loss = criterion(output.abs(), gnd.abs(), fg_mask)
                    
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)

                # apply hard constraints
                for p in net.parameters():
                    if p.requires_grad and hasattr(p, 'proj'):
                        p.proj()

                global_step += 1
                epoch_loss += loss.item()

                log_input_im = x0[0].abs().flip(1)
                log_output_im = output[0].abs().flip(1)
                log_gnd_im = gnd[0].abs().flip(1)
                log_fg_mask_im = fg_mask[0].flip(1)
                log_im_list = [
                    log_input_im * log_fg_mask_im / log_gnd_im.max(),
                    log_output_im * log_fg_mask_im / log_gnd_im.max(),
                    log_gnd_im * log_fg_mask_im / log_gnd_im.max(),
                ]
                if args.aleatoric:
                    log_aleatoric = aleatoric_std[0].flip(1)   
                    log_im_list += [log_aleatoric * log_fg_mask_im / log_aleatoric.max()]

                log_im = torch.cat(log_im_list, dim=2)
                log_im = torch.clamp_max(log_im, 1)


                log_dict = {
                    'train/loss': loss.item(),
                    #'train/step': global_step,
                    #'train/batch': global_step,
                    'train/epoch': epoch,
                    # 'train/max_intensity': log_gnd_im.max(),
                    'train/in_out_gnd': wandb.Image(log_im.float().cpu()),
                    'train/lr': optimizer.param_groups[0]['lr']
                }


                # if args.epistemic:
                #     log_dict['train/max_epistemic'] = log_aleatoric.max()
                if global_step % 200 == 0:
                    log_dict['train/in_out_gnd'] = wandb.Image(log_im.float().cpu())


                experiment.log(log_dict, step=global_step)
                pbar.update(x0.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            avg_epoch_loss = epoch_loss / n_train * config['batch_size']
            logging.info('Avg Loss of Epoch {}: {}'.format(epoch, avg_epoch_loss))

        # evaluation
        histograms = {}
        for tag, value in net.named_parameters():
            tag = tag.replace('/', '.')
            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            if value.requires_grad:
                histograms['Gradients' + tag] = wandb.Histogram(value.grad.data.cpu())
        
        

        net.eval()
        #val_score = 0
        val_l1 = 0
        val_nmse = 0
        val_psnr = 0
        val_ssim = 0
        with torch.no_grad():
            #val_tab = wandb.Table()
            #val_tab.add_row("image", "l1", "nmse", "psnr", "ssim")
            with tqdm(total=n_val, desc=f'Val Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar_val:
                for idx, batch in enumerate(val_dataloader):
                    inputs, outputs = fastmri.data.prepare_batch(batch, device)

                    x0 = inputs[0]
                    fg_mask = inputs[-1] if config['use_fg_mask'] else torch.ones_like(x0, dtype=torch.float32)
                    gnd = outputs[0]

                    if config['use_fg_mask']:
                        selected_input = inputs[:-1] # pass all inputs except fg_mask
                    else:
                        selected_input = inputs

                    if args.epistemic:
                        output_samples_raw = [net(*selected_input) for _ in range(args.num_samples)]
                        if args.aleatoric:
                            output_samples_raw, raw_uncertainty_samples = zip(*output_samples_raw)
                            raw_uncertainty = torch.mean(torch.stack(raw_uncertainty_samples), axis=0)

                        output_samples = torch.stack(output_samples_raw)                    
                        output = torch.mean(output_samples, axis=0)
                        epistemic_var = torch.var(output_samples, axis=0)
                        # epistemic_var = torch.real(epistemic_var_complex) + torch.imag(epistemic_var_complex)
                    else:
                        output_raw = net(*selected_input)
                        if args.aleatoric:
                            output, raw_uncertainty = output_raw                        
                            # val_loss = criterion(output.abs(), gnd.abs(), fg_mask, raw_uncertainty)

                        else:
                            output = output_raw
                            # val_loss = criterion(output.abs(), gnd.abs(), fg_mask)
                            pass

                    if args.aleatoric:
                        if args.l2:
                            aleatoric_std = torch.sqrt(torch.exp(raw_uncertainty))
                        else:
                            aleatoric_std = torch.exp(raw_uncertainty) * torch.sqrt(raw_uncertainty.new_tensor(2))


                    #val_loss = F_fastmri.masked_l1_loss(output.abs(), gnd.abs(), fg_mask)
                    l1, nmse, psnr, ssim = F_fastmri.evaluate(output.abs(), gnd.abs(), (-2,-1), fg_mask)

                    # logging.info('Validation {}/{} loss: {}'.format(idx, ceil(n_val / args.batch_size), val_loss))

                    log_input_im = x0[0].abs().flip(1)
                    log_output_im = output[0].abs().flip(1)
                    log_gnd_im = gnd[0].abs().flip(1)
                    log_fg_mask_im = fg_mask[0].flip(1)

                    log_im_list = [
                        log_input_im * log_fg_mask_im / log_gnd_im.max(),
                        log_output_im * log_fg_mask_im / log_gnd_im.max(),
                        log_gnd_im * log_fg_mask_im / log_gnd_im.max(),
                    ]
                    if args.aleatoric:
                        log_aleatoric = aleatoric_std[0].flip(1)
                        log_im_list += [log_aleatoric * log_fg_mask_im / log_aleatoric.max()]

                    if args.epistemic:
                        log_epistemic = torch.sqrt(epistemic_var[0]).flip(1)
                        log_im_list += [log_epistemic * log_fg_mask_im / log_epistemic.max()]

                    log_im = torch.cat(log_im_list, dim=2) 
                    log_im = torch.clamp_max(log_im, 1)

                    #val_score += val_loss / len(val_dataloader)
                    val_l1 += l1 / len(val_dataloader)
                    val_nmse += nmse / len(val_dataloader)
                    val_psnr += psnr / len(val_dataloader)
                    val_ssim += ssim / len(val_dataloader)
                    
                    #val_tab.add_row(wandb.Image(log_im.float().cpu()), )

                    pbar_val.update(x0.shape[0])
                    # pbar_val.set_postfix(**{'loss (batch)': l1.item()})
                    # experiment.log({
                    #     'val_images/results': wandb.Image(log_im.float().cpu()),
                    #     'val_imaegs/idx': idx
                    # })


            log_dict = {
                #'val/lr': optimizer.param_groups[0]['lr'],
                'val/loss': val_l1,
                'val/nmse': val_nmse,
                'val/psnr': val_psnr,
                'val/ssim': val_ssim,
                'val/results': wandb.Image(log_im.float().cpu()),
                #'trian/epoch': epoch + 1,
                #'val/step': global_step,
                #'val/step': epoch + 1,
                'val/epoch': epoch,
                **histograms
            }

            # if args.aleatoric:
            #     log_dict['val/max_aleatoric'] = log_aleatoric.max()
            # if args.epistemic:
            #     log_dict['val/max_epistemic'] = log_aleatoric.max()

            experiment.log(log_dict, step=global_step)
            pbar_val.update(x0.shape[0])
            # pbar_val.set_postfix(**{'loss (batch)': loss.item()})
            val_score = val_l1
            scheduler.step(val_score)



        if args.save_checkpoint:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
            if epoch == 0:
                import json
                with open(Path(checkpoint_path) / 'config.json', 'w') as fp:
                    json.dump(vars(args), fp, indent=4, sort_keys=True)
            torch.save(net.state_dict(), str(checkpoint_path / 'checkpoint_epoch{}.pth'.format(epoch+1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.info(args)
    
    aleatoric = (("combined" if args.combined_aleatoric else "separate") if args.aleatoric else None)

    if args.model == 'dncn':
        net = DnCn(regularizer=args.regularizer, shared_params=args.shared_params, nc=args.nc, nf=args.nf, dropout_probability=args.dropout, aleatoric=aleatoric, epistemic=args.epistemic)
    elif args.model == 'unet':
        net = Unet(in_chans=2, out_chans=2, chans=32, num_pool_layers=4)
    elif args.model == 'snet':
        net = Snet(n_blocks=10, input_dim=2, n_f=64)
    # net = DnCnComplexDP()
    logging.info(f'Network initialized!')
    logging.info(net)
    
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

    
