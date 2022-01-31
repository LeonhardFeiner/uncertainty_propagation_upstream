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
from fastmri.data.fastmri_dataloader_th import FastmriCartesianDataset
import fastmri.functional as F_fastmri
from pathlib import Path
import pickle
import scipy.io as scio

def get_args():
    parser = ArgumentParser()

    # training related
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--mode", default='singlecoil_val')
    parser.add_argument("--cal_metrics", default=False)
    parser.add_argument("--wandb", default=False)
    parser.add_argument("--save_mat", default=False)
    parser.add_argument("--save_pickle", default=False)

    parser.add_argument("--model", default='dncn', type=str)
    parser.add_argument("--num_workers", default=8, type=str)
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
    

def test_all(net, device, args):
    config_path = './config.yml'
    config = loadYaml(config_path, 'BaseExperiment')
    config['batch_size'] = 1
    dataset = FastmriCartesianDataset(config=config, mode=args.mode, extra_keys=["fname"])
    loader_args = dict(batch_size=1, num_workers=args.num_workers, pin_memory=True)
    dataloader = DataLoader(dataset, shuffle=False, **loader_args)
    
    exp_id = args.ckpt_path.split('/')[-2]

    if args.save_mat or args.save_pickle:
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
                inputs, outputs, (filename,) = fastmri.data.prepare_batch(batch, device)

                x0 = inputs[0]
                gnd = outputs[0]

                if config['use_fg_mask']:
                    selected_input = inputs[:-1] # pass all inputs except fg_mask
                    fg_mask = inputs[-1]

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
                else:
                    output_raw = net(*selected_input)
                    if args.aleatoric:
                        output, raw_uncertainty = output_raw                        

                    else:
                        output = output_raw
                        pass

                if args.aleatoric:
                    if args.l2:
                        aleatoric_std = torch.sqrt(torch.exp(raw_uncertainty))
                    else:
                        aleatoric_std = torch.exp(raw_uncertainty) * torch.sqrt(raw_uncertainty.new_tensor(2))

                if config['use_fg_mask']:
                    output *= fg_mask
                else:
                    fg_mask = torch.ones_like(output.abs())

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
                    log_im_list = [
                        log_input_im * log_fg_mask_im, # / log_gnd_im.max(),
                        log_output_im * log_fg_mask_im,# / log_gnd_im.max(),
                        log_gnd_im * log_fg_mask_im,# / log_gnd_im.max(),
                    ]
                    if args.aleatoric:
                        log_aleatoric = aleatoric_std[0].flip(1)   
                        log_im_list += [log_aleatoric * log_fg_mask_im] # / log_aleatoric.max()]

                    if args.epistemic:
                        log_epistemic = torch.sqrt(epistemic_var[0]).flip(1)
                        log_im_list += [log_epistemic * log_fg_mask_im] # / log_epistemic.max()

                    log_im = torch.cat(log_im_list, dim=2)

                    log_im = torch.clamp_max(log_im, 1)
                    
                    if args.wandb:
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

                if args.save_mat or args.save_pickle:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    result_dict =  {'recon':output, 'gnd': gnd}
                    if args.aleatoric:
                        result_dict['aleatoric'] = torch.square(aleatoric_std)
                    if args.epistemic:
                        result_dict['epistemic'] = epistemic_var
                        
                    name = Path(filename[0]).stem

                if args.save_mat:
                    scio.savemat(save_dir / ('test_' + name + '.mat'), result_dict)
                if args.save_pickle:
                    result_dict = {key: value.cpu().numpy() for key, value in result_dict.items()}
                    with open(save_dir / ('test_' + name + '.pickle'), "wb") as file:
                        pickle.dump(result_dict, file)

                pbar.update(x0.shape[0])
        
    if args.cal_metrics:
        if args.wandb:
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

    aleatoric = (("combined" if args.combined_aleatoric else "separate") if args.aleatoric else None)

    if args.model == 'dncn':
        net = DnCn(regularizer=args.regularizer, shared_params=args.shared_params, nc=args.nc, nf=args.nf, dropout_probability=args.dropout, aleatoric=aleatoric, epistemic=args.epistemic)
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