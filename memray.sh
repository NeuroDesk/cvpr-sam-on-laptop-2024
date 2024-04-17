#!/bin/bash
python3 -m memray run CVPR24_LiteMedSamOnnx_infer.py -i /workspace/inputs -o /workspace/outputs
mv memray-*.bin /workspace/outputs