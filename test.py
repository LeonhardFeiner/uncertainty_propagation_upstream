from argparse import ArgumentParser
import logging
import sys
import numpy as np
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
    parser.add_argument("--cal-metrics", action='store_true')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--save-mat", action='store_true')
    parser.add_argument("--save-pickle", action='store_true')
    parser.add_argument("--save-numpy", action='store_true')
    parser.add_argument("--save-gnd", action='store_true')
    parser.add_argument('--no-save-gnd', dest='save_gnd', action='store_false')
    parser.set_defaults(save_gnd=True)

    parser.add_argument("--model", default='dncn', type=str)
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
    

def test_all(net, device, args):
    config_path = './config.yml'
    config = loadYaml(config_path, 'BaseExperiment')
    # config['batch_size'] = 1
    dataset = FastmriCartesianDataset(config=config, mode=args.mode, extra_keys=["fname", "slidx"])
    loader_args = dict(batch_size=1, num_workers=args.num_workers, pin_memory=True)
    dataloader = DataLoader(dataset, shuffle=False, **loader_args)
    
    exp_id = args.ckpt_path.split('/')[-2]

    if args.save_mat or args.save_pickle or args.save_numpy:
        acc_str = "_".join(f"{acc:02}" for acc in config["accelerations"])
        save_dir = Path('./test_results') / f"{config['dataset_name']}_{args.mode}" / f"{exp_id}_acc{acc_str}" 
        print(save_dir)
    n_test = len(dataset)

    #total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    if args.wandb:
        expconfig = vars(args)
        #expconfig["total_params"] = total_params
        experiment = wandb.init(project="dccnn-dp", reinit=False, name='TEST-'+exp_id, config=expconfig)

    test_idx = 0
    sample_count = 0
    avg_l1 = 0
    avg_l2 = 0
    if args.aleatoric:
        avg_aleatoric = 0
    if args.epistemic:
        avg_epistemic = 0
    avg_nmse = 0
    avg_psnr = 0
    avg_ssim = 0
    with torch.no_grad():
        with tqdm(total=n_test, desc=f'Test case {test_idx + 1}/{n_test}', unit='img') as pbar:
            last_filename = None
            files_count = 0
            for batch in dataloader:
                test_idx += 1
                inputs, outputs, (filename,), slidx = fastmri.data.prepare_batch(batch, device)

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
                    l1, l2, nmse, psnr, ssim = F_fastmri.evaluate(output.abs(), gnd.abs(), (-2,-1), fg_mask)
                    
                    if args.aleatoric:
                        avg_aleatoric += torch.mean(torch.square(aleatoric_std)) * len(x0)
                    if args.epistemic:
                        avg_epistemic += torch.mean(epistemic_var) * len(x0)

                    sample_count += len(x0)
                    avg_l1 += l1 * len(x0)
                    avg_l2 += l2 * len(x0)
                    avg_nmse += nmse * len(x0)
                    avg_psnr += psnr * len(x0)
                    avg_ssim += ssim * len(x0)

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

                if args.save_mat or args.save_pickle or args.save_numpy:    
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    result_dict =  {'recon':output, "slidx": torch.as_tensor(slidx)}

                    if args.save_gnd:
                        result_dict['gnd'] =  gnd
                    if args.aleatoric:
                        result_dict['aleatoric_std'] = aleatoric_std
                    if args.epistemic:
                        result_dict['epistemic_std'] = torch.sqrt(epistemic_var)
                        
                    if last_filename == filename:
                        full_result_dict = {
                            key: torch.concat((last, result_dict[key].cpu()))
                            for key, last in full_result_dict.items()
                        }
                    else:
                        if last_filename is None:
                            pbar.write("keys to save: " +",".join(result_dict.keys()))
                        if last_filename is not None:
                            name = Path(last_filename).stem
                            full_result_dict = {key: value.numpy() for key, value in full_result_dict.items()}
                            files_count += 1

                            mml = [str(fn(full_result_dict["slidx"])) for fn in (min, lambda x: max(x) + 1, len)]
                            pbar.set_postfix(file_index=files_count, slice_range=":".join(mml))

                            if args.save_mat:
                                scio.savemat(save_dir / ('test_' + name + '.mat'), full_result_dict)                            
                            if args.save_pickle:
                                with open(save_dir / ('test_' + name + '.pickle'), "wb") as file:
                                    pickle.dump(full_result_dict, file)
                            if args.save_numpy:
                                np.savez_compressed(save_dir / ('test_' + name + '.npz'), **full_result_dict)
                        
                        full_result_dict = {key: value.cpu() for key, value in result_dict.items()}
                last_filename = filename
                pbar.update(1)

            name = Path(last_filename).stem
            if args.save_mat:
                scio.savemat(save_dir / ('test_' + name + '.mat'), full_result_dict)
            if args.save_pickle:
                full_result_dict = {key: value.numpy() for key, value in full_result_dict.items()}
                with open(save_dir / ('test_' + name + '.pickle'), "wb") as file:
                    pickle.dump(full_result_dict, file)
        
    if args.cal_metrics:

        if args.aleatoric:
            avg_aleatoric /= sample_count
            avg_aleatoric_std = torch.sqrt(avg_aleatoric)
        if args.epistemic:
            avg_epistemic /= sample_count
            avg_epistemic_std = torch.sqrt(avg_epistemic)


        avg_l1 /= sample_count
        avg_l2 /= sample_count
        avg_nmse /= sample_count
        avg_psnr /= sample_count
        avg_ssim /= sample_count

        log_dict = {
                'avg_l1': avg_l1.item(),
                'avg_l2': avg_l2.item(),
                'avg_nmse': avg_nmse.item(),
                'avg_psnr': avg_psnr.item(),
                'avg_ssim': avg_ssim.item(),
            }

        if args.aleatoric:
            log_dict['avg_aleatoric_std'] = avg_aleatoric_std.item()
        if args.epistemic:
            log_dict['avg_epistemic_std'] = avg_epistemic_std.item()

        if args.wandb:
            experiment.log(log_dict)

        for key, value in log_dict.items():
            logging.info(f'{key}: {value}')

                


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