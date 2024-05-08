# %% move files based on csv file
import numpy as np
import nibabel as nb
import os
from os import listdir, makedirs
from os.path import basename, join, dirname, isfile, isdir
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import multiprocessing as mp
import argparse
import matplotlib.pyplot as plt

def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in range(1, gt.max()+1):
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.mean(dsc)

def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in range(1, gt.max()+1):
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing
        )
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    return np.mean(nsd)


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seg_dir', default='./test_demo/segs', type=str)
parser.add_argument('-g', '--gt_dir', default='./test_demo/gts', type=str)
parser.add_argument('-o', '--overlay_dir', default='./test_demo/overlay', type=str)
parser.add_argument('-csv_dir', default='./test_demo/metrics.csv', type=str)
parser.add_argument('-num_workers', type=int, default=5)
parser.add_argument('-nsd', default=True, type=bool, help='set it to False to disable NSD computation and save time')
parser.add_argument('-wandb_log', default=False, action='store_true', help='set it to True to enable wandb dashboard')
parser.add_argument('-wandb_key', default='test_demo_python_litemedsam', type=str)
args = parser.parse_args()

seg_dir = args.seg_dir
gt_dir = args.gt_dir
overlays_available = False
if os.path.isdir(args.overlay_dir):
    overlays_available = True
    overlay_dir = args.overlay_dir
else:
    print(f"Warning: Overlay directory {args.overlay_dir} does not exist")
csv_dir = args.csv_dir
num_workers = args.num_workers
compute_NSD = args.nsd
wandb_log = args.wandb_log
wandb_key = args.wandb_key

def compute_metrics(npz_name):
    metric_dict = {'dsc': -1.}
    if compute_NSD:
        metric_dict['nsd'] = -1.
    npz_seg = np.load(join(seg_dir, npz_name), allow_pickle=True, mmap_mode='r')
    npz_gt = np.load(join(gt_dir, npz_name), allow_pickle=True, mmap_mode='r')
    gts = npz_gt['gts']
    segs = npz_seg['segs']
    if npz_name.startswith('3D'):
        spacing = npz_gt['spacing']
    
    dsc = compute_multi_class_dsc(gts, segs)
    # comupute nsd
    if compute_NSD:
        if dsc > 0.2:
        # only compute nsd when dice > 0.2 because NSD is also low when dice is too low
            if npz_name.startswith('3D'):
                nsd = compute_multi_class_nsd(gts, segs, spacing)
            else:
                spacing = [1.0, 1.0, 1.0]
                nsd = compute_multi_class_nsd(np.expand_dims(gts, -1), np.expand_dims(segs, -1), spacing)
        else:
            nsd = 0.0
    return npz_name, dsc, nsd, gts.shape

if __name__ == '__main__':
    seg_metrics = OrderedDict()
    seg_metrics['case'] = []
    seg_metrics['modality'] = []
    seg_metrics['image size'] = []
    seg_metrics['dsc'] = []
    if compute_NSD:
        seg_metrics['nsd'] = []
    
    npz_names = listdir(gt_dir)
    npz_names = [npz_name for npz_name in npz_names if npz_name.endswith('.npz')]
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(npz_names)) as pbar:
            for i, (npz_name, dsc, nsd, shape) in enumerate(pool.imap_unordered(compute_metrics, npz_names)):
                seg_metrics['case'].append(npz_name)
                modality = npz_name.split('_')[1]
                seg_metrics['modality'].append(modality)
                seg_metrics['image size'].append(shape)
                seg_metrics['dsc'].append(np.round(dsc, 4))
                if compute_NSD:
                    seg_metrics['nsd'].append(np.round(nsd, 4))
                pbar.update()
    seg_metrics['case'].append('Mean')
    seg_metrics['modality'].append('N/A')
    seg_metrics['image size'].append('N/A')
    seg_metrics['dsc'].append(np.round(np.mean(seg_metrics['dsc']), 4))
    if compute_NSD:
        seg_metrics['nsd'].append(np.round(np.mean(seg_metrics['nsd']), 4))
    df = pd.DataFrame.from_dict(seg_metrics)
    # rank based on case column
    df = df.sort_values(by=['case'])
    df.to_csv(csv_dir, index=False)
    print("Metrics data saved to:", csv_dir)

    if wandb_log:
        import wandb
        wandb.init(
            project='MedSAM_Laptop',  
            name=wandb_key,
            )
        wandb_table = OrderedDict()
        wandb_table['Modality'] = []
        wandb_table['DSC mean'] = []
        wandb_table['DSC std'] = []
        if compute_NSD:
            wandb_table['NSD mean'] = []
            wandb_table['NSD std'] = []
        if overlays_available:
            wandb_table['Worse case'] = []
    # metrics group by modality
    modality_group = df.groupby('modality')
    # min_case = modality_group['dsc'].idxmin()
    for i, (modality, group) in enumerate(modality_group):
        dsc_mean, dsc_std = group['dsc'].mean(), group['dsc'].std()
        nsd_mean, nsd_std = group['nsd'].mean(), group['nsd'].std()
        # print(f'{modality}: DSC mean: {dsc_mean:.4f}, DSC std: {dsc_std:.4f}')
        # if compute_NSD:
        #     print(f'{modality}: NSD mean: {nsd_mean:.4f}, NSD std: {nsd_std:.4f}')      
        if wandb_log:
            import wandb
            wandb_table['Modality'].append(modality)
            wandb_table['DSC mean'].append(dsc_mean)
            wandb_table['DSC std'].append(dsc_std)
            if compute_NSD:
                wandb_table['NSD mean'].append(nsd_mean)
                wandb_table['NSD std'].append(nsd_std)
            # minimum case for each modality
            if overlays_available:
                overlay_id = df.iloc[group['dsc'].idxmin()]['case'].split('.')[0]
                overlay = plt.imread(os.path.join(overlay_dir, f'{overlay_id}.png'))  
                wandb_table['Worse case'].append(wandb.Image(overlay, caption=overlay_id))
            
    if wandb_log:
        import wandb
        print(wandb_table)
        wandb_df = pd.DataFrame(wandb_table)

        dsc_mean, dsc_std = df['dsc'].mean(), df['dsc'].std()
        nsd_mean, nsd_std = df['nsd'].mean(), df['nsd'].std()
        wandb_mean = {'Modality':'Overall',
                    'DSC mean': dsc_mean, 
                    'DSC std': dsc_std,
                    'NSD mean': nsd_mean,
                    'NSD std': nsd_std}
        df = pd.concat([pd.DataFrame(wandb_mean, index=[0]), wandb_df])
        table = wandb.Table(dataframe=df)
        wandb.log({wandb_key: table})
        
