#!/bin/bash --login
#SBATCH --nodes 1
#SBATCH --job-name=litemedsam
#SBATCH --ntasks 1
#SBATCH -c 30
#SBATCH --mem=50000
#SBATCH -o logs/litemedsam_sharp_out.txt
#SBATCH -e logs/litemedsam_sharp_error.txt
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --account=a_barth
#SBATCH --time=72:00:00

module load cuda


source activate medsam

srun python train_encoder_one_gpu_sharp_lora.py -data_root /QRISdata/Q7010/datasets/train_npy \
    -pretrained_checkpoint /QRISdata/Q7010/checkpoints/LiteMedSAM/lite_medsam.pth \
    -work_dir work_dir/pet_microscope \
    -num_epochs 50 \
    -batch_size 256 \