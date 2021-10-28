from argparse import ArgumentParser
import logging
import sys
from numpy.lib.npyio import save
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import wandb
import fastmri.data
from fastmri.models.dncn import DnCn
from fastmri.models.unet import Unet
from fastmri.models.snet import Snet
from fastmri.utils import loadYaml
import fastmri_dataloader
from fastmri.data.fastmri_dataloader_th import FastmriCartesianDataset
import fastmri.functional as F_fastmri
from pathlib import Path
import scipy.io as scio

def get_args():
    parser = ArgumentParser()

    # training related
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--mode", default='singlecoil_val')
    parser.add_argument("--cal_metrics", default=False)
    parser.add_argument("--wandb", default=False)
    parser.add_argument("--save_mat", default=False)

    parser.add_argument("--model", default='dncn', type=str)
    parser.add_argument("--num_workers", default=8, type=str)
    parser.add_argument("--regularizer", default='Real2chCNN', type=str)
    parser.add_argument("--shared-params", action='store_true', help='Sharing paramters over cascades')
    parser.add_argument("--nc", type=int, default=10, help='Number of cascades')
    parser.add_argument("--nf", type=int, default=64, help='Num base features')

    return parser.parse_args()
    

def test_all(net, device, args):
    config_path = './config.yml'
    config = loadYaml(config_path, 'BaseExperiment')
    config['batch_size'] = 1
    dataset = FastmriCartesianDataset(config=config, mode=args.mode)
    loader_args = dict(batch_size=1, num_workers=8, pin_memory=True)
    dataloader = DataLoader(dataset, shuffle=False, **loader_args)
    
    exp_id = args.ckpt_path.split('/')[-2]

    if args.save_mat:
        save_dir = Path('./test_results') / exp_id

    n_test = len(dataset)

    #total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    if args.wandb:
        expconfig = vars(args)
        #expconfig["total_params"] = total_params
        experiment = wandb.init(project="dccnn-dp", reinit=False, name='TEST-'+exp_id, config=expconfig)

    test_idx = 0
    avg_l1 = 0
    avg_nmse = 0
    avg_psnr = 0
    avg_ssim = 0
    with torch.no_grad():
        with tqdm(total=n_test, desc=f'Test case {test_idx + 1}/{n_test}', unit='img') as pbar:
            for batch in dataloader:
                test_idx += 1
                inputs, outputs = fastmri.data.prepare_batch(batch, device)

                x0 = inputs[0]
                fg_mask = inputs[-1]
                gnd = outputs[0]

                output = net(*inputs[:-1]) # pass all inputs except fg_mask
                output *= fg_mask

                if args.cal_metrics:
                    l1, nmse, psnr, ssim = F_fastmri.evaluate(output.abs(), gnd.abs(), (-2,-1), fg_mask)

                avg_l1 += l1 / len(dataloader)
                avg_nmse += nmse / len(dataloader)
                avg_psnr += psnr / len(dataloader)
                avg_ssim += ssim / len(dataloader)

                if args.wandb:
                    log_input_im = x0[0].abs().flip(1)
                    log_output_im = output[0].abs().flip(1)
                    log_gnd_im = gnd[0].abs().flip(1)
                    log_fg_mask_im = fg_mask[0].flip(1)

                    log_im = torch.cat([log_input_im * log_fg_mask_im,
                                        log_output_im * log_fg_mask_im,
                                        log_gnd_im * log_fg_mask_im
                                        ], dim=2) / log_gnd_im.max()
                    log_im = torch.clamp_max(log_im, 1)
                    
                    if args.cal_metrics:
                        experiment.log({
                            'val/loss': l1,
                            'val/nmse': nmse,
                            'val/psnr': psnr,
                            'val/ssim': ssim,
                            'val/results': wandb.Image(log_im.float().cpu()),
                        })
                    else:
                        experiment.log({
                            'val/results': wandb.Image(log_im.float().cpu()),
                        })

                if args.save_mat:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    scio.savemat(save_dir / ('test_' + str(test_idx) + '.mat'), {'recon':output, 'gnd': gnd})

                pbar.update(x0.shape[0])
        
    if args.cal_metrics:
        experiment.log({
            'avg_nmse': avg_nmse,
            'avg_l1': avg_l1,
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim
        })
        logging.info(f'avg_l1: {avg_l1}')
        logging.info(f'avg_nmse: {avg_nmse}')
        logging.info(f'avg_psnr: {avg_psnr}')
        logging.info(f'avg_ssim: {avg_ssim}')
                


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.info(args)

    if args.model == 'dncn':
        net = DnCn(regularizer=args.regularizer, shared_params=args.shared_params, nc=args.nc, nf=args.nf)
    elif args.model == 'unet':
        net = Unet(in_chans=2, out_chans=2, chans=32, num_pool_layers=4)
    elif args.model == 'snet':
        net = Snet(n_blocks=10, input_dim=2, n_f=64)

    net.load_state_dict(torch.load(args.ckpt_path))
    net.to(device)
    net.eval()

    logging.info(f'Network initialized. Weight loaded.')
    logging.info(net)

    try:
        test_all(net, device, args)
    except KeyboardInterrupt:
        logging.info(f'User interrupt')
        sys.exit(0)