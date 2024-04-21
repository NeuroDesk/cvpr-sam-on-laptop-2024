#!/bin/bash

# This script runs inference on all the test images, 
# and outputs running time and metrics.
# This script should be run from the repo root.
# Since running docker may require root privileges, 
# the sudo password should be piped into the stdin of this script

metrics_save_path="$1"
timestamp=$(date +%d-%m-%Y-%H-%M-%S)

run_dir=test_demo/runs/"$timestamp"
mkdir -p "$run_dir"

# If the path to save metrics (including running time) is not given
# then save them in the directory created for this run
if [[ -z "$1" ]]; then
    metrics_save_path="$run_dir"
fi

python3 CVPR24_time_eval.py -n "$DOCKER_IMAGE_NAME" -i ./test_demo/imgs \
    -o ./test_demo/runs/"$timestamp"/segs \
    --timing_save_path "$metrics_save_path"
python3 evaluation/compute_metrics.py -s ./test_demo/runs/"$timestamp"/segs \
    -g test_demo/gts -csv_dir "$metrics_save_path"/metrics.csv