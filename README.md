# [Modality-Specific Strategies for Medical Image Segmentation using Lightweight SAM Architectures](https://openreview.net/forum?id=bEQ2KJ9Cgw)  
 
<img src="assets\figure1-methods.png" alt="method" width="700"/>  

This repository is our approach to the challenge [CVPR 2024: SEGMENT ANYTHING IN MEDICAL IMAGES ON LAPTOP](https://www.codabench.org/competitions/1847/#/pages-tab). This challenge focuses on efficiently segmenting multi-modality medical image using lightweight bounding box-based segmentation model on CPU.


## Evaluation:

Segmentation accuracy metrics:

- Dice Similarity Coefficient (DSC)
- Normalized Surface Distance (NSD)

Segmentation efficiency metrics:

- Running time (s)

## Method:

Pretrained models and publicly available datasets (registered before April 15, 2024)

- Pretrain Models other than SAM: https://paperswithcode.com/sota/interactive-segmentation-on-grabcut?p=reviving-iterative-training-with-mask
- Public Dataset: ?

Hardware specs:

- CPU: Intel® Xeon(R) W-2133 @ 3.60GHz RAM: 8G
- Docker version 20.10.13

## Challeng rule agreement consent

https://drive.google.com/file/d/1XnrKAntAwZo3neEEMNBrKU0h4udoN64h/view

### Installation

The codebase is tested with: `Ubuntu 20.04` | Python `3.10` | `CUDA 11.8` | `Pytorch 2.1.2`

1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/NeuroDesk/cvpr-sam-on-laptop-2024.git`
4. Enter the MedSAM folder `cd cvpr-sam-on-laptop-2024` and run `pip install -e .`

### Quick tutorial on making submissions to CVPR 2024 MedSAM on Laptop Challenge

#### Sanity test

- Download the LiteMedSAM, finetuned LiteMedSAM, EfficientSAM onnx models [here](https://files.au-1.osf.io/v1/resources/u8tny/providers/osfstorage/6618c57de65c6053727d9cbf/?zip=) and put it in `work_dir/onnx_models`. After unzipping, the directory structure should be following. The LiteMedSAM_finetuned_PET_Mircoscope folder should be renamed as LiteMedSAM_finetuned and the LiteMedSAM folder should be renamed as LiteMedSAM_preprocess. The model files within them should be renamed as shown below.

```
work_dir/
├── onnx_models
│   ├── EfficientSAM
│   │   ├── efficient_sam_vitt_decoder.quant.onnx
│   │   └── efficient_sam_vitt_encoder.quant.onnx
│   ├── LiteMedSAM_finetuned
│   │   ├── litemedsam_decoder.onnx
│   │   └── litemedsam_encoder.onnx
│   └── LiteMedSAM_preprocess
│       ├── litemedsam_decoder.onnx
│       └── litemedsam_encoder.onnx
```

- Download the demo data [here](https://drive.google.com/drive/folders/1t3Rs9QbfGSEv2fIFlk8vi7jc0SclD1cq?usp=sharing). The directory structure should be following.

```
test_demo/
├── gts
│   ├── *.npz
│   └── *.npz
├── imgs
│   ├── *.npz
│   └── *.npz
└── segs
    ├── *.npz
    └── *.npz
```
- Run the following command for a sanity test using ONNX models

```bash
python CVPR24_LiteMedSamOnnx_infer.py -i test_demo/imgs/ -o test_demo/segs
```

#### Build Docker

```bash
docker build -f Dockerfile -t hawken50 .
```

> Note: don't forget the `.` in the end

Run the docker on the testing demo images

```bash
docker container run -m 8G --name hawken50 --rm -v $PWD/test_demo/imgs/:/workspace/inputs/ -v $PWD/test_demo/hawken50/:/workspace/outputs/ hawken50:latest /bin/bash -c "sh predict.sh"
```

> Note: please run `chmod -R 777 ./*` if you run into `Permission denied` error.

Save docker

```bash
docker save hawken50 | gzip -c > hawken50.tar.gz
```

#### Compute Metrics

```bash
python evaluation/compute_metrics.py -s test_demo/hawken50 -g test_demo/gts -csv_dir ./metrics.csv
```

#### Export ONNX model
Download the LiteMedSAM, finetuned LiteMedSAM, EfficientSAM checkpoints [here](https://files.au-1.osf.io/v1/resources/u8tny/providers/osfstorage/6649998e915ae40b30e8993a/?zip=) and put it in `work_dir/checkpoints`. 
Change `--checkpoint` and `--model-type` argument to export from different checkpoints where `vit_h`, `vit_l`, `vit_b`, `vit_t` for LiteMedSAM and `vitt`, `vits` for EfficientSAM.

```bash
python onnx_decoder_exporter.py --checkpoint work_dir/checkpoints/lite_medsam.pth --output work_dir/onnx_models/lite_medsam_encoder.onnx --model-type vit_t --return-single-mask
```

### Model Training

#### Fine-tune pretrained Lite-MedSAM

> The training pipeline requires about 10GB GPU memory with a batch size of 4

##### Single GPU

To train Lite-MedSAM on a single GPU, run:

```bash
python train_one_gpu.py \
    -data_root data/MedSAM_train \
    -pretrained_checkpoint lite_medsam.pth \
    -work_dir work_dir \
    -num_workers 4 \
    -batch_size 4 \
    -num_epochs 10
```

To resume interrupted training from a checkpoint, run:

```bash
python train_one_gpu.py \
    -data_root data/MedSAM_train \
    -resume work_dir/medsam_lite_latest.pth \
    -work_dir work_dir \
    -num_workers 4 \
    -batch_size 4 \
    -num_epochs 10
```

For additional command line arguments, see `python train_one_gpu.py -h`.

##### Multi-GPU

To fine-tune Lite-MedSAM on multiple GPUs, run:

```bash
python train_multi_gpus.py \
    -i data/npy \ ## path to the training dataset
    -task_name MedSAM-Lite-Box \
    -pretrained_checkpoint lite_medsam.pth \
    -work_dir ./work_dir_ddp \
    -batch_size 16 \
    -num_workers 8 \
    -lr 0.0005 \
    --data_aug \ ## use data augmentation
    -world_size <WORLD_SIZE> \ ## Total number of GPUs will be used
    -node_rank 0 \ ## if training on a single machine, set to 0
    -init_method tcp://<MASTER_ADDR>:<MASTER_PORT>
```

Alternatively, you can use the provided `train_multi_gpus.sh` script to train on multiple GPUs. To resume interrupted training from a checkpoint, add `-resume <your_work_dir>` to the command line arguments instead of the checkpoint path for multi-GPU training;
the script will automatically find the latest checkpoint in the work directory. For additional command line arguments, see `python train_multi_gpus.py -h`.

### Inference using Pytorch model checkpoint
To run inference using modality specific strategy, we need three model checkpoints. Download the LiteMedSAM, finetuned LiteMedSAM, EfficientSAM checkpoints [here](https://files.au-1.osf.io/v1/resources/u8tny/providers/osfstorage/6649998e915ae40b30e8993a/?zip=) and put it in `work_dir/checkpoints`. 

```bash
python CVPR24_infer.py \
    --data_root test_demo/imgs \
    --pred_save_dir segs_python \
    --save_overlay True \
    --png_save_dir overlay_python \
    --model_name litemedsam \
    --checkpoint_path work_dir/checkpoints/lite_medsam.pth \
    --efficientsam_path work_dir/checkpoints/efficient_sam_vitt.pt \
    --finetuned_pet_path work_dir/checkpoints/medsam_lite_encoder_pet_micro_sharp_epoch50_lr0.00005.pth
```

 





### Acknowledgements

We thank the authors of [LiteMedSAM](https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM) and [EfficientSAM](https://github.com/yformer/EfficientSAM) for making their source code publicly available.
