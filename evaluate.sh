#!/bin/bash

# This script runs inference on all the test images, 
# and outputs running time and metrics.
# This script should be run from the repo root.
# Since running docker may require root privileges, 
# the sudo password should be piped into the stdin of this script
# 
# Prerequisites: The test data should be downloaded
# and placed in the correct folder according to the repo README
# The model checkpoints should be in work_dir/

metrics_save_path="$1"
timestamp=$(date +%d-%m-%Y-%H-%M-%S)

run_dir=test_demo/runs/"$timestamp"
mkdir -p "$run_dir"

# If the path to save metrics (including running time) is not given
# then save them in the directory created for this run
if [[ -z "$1" ]]; then
    metrics_save_path="$run_dir"
fi

# Check that the DOCKER_IMAGE_NAME env variable is set
if [[ -z "${DOCKER_IMAGE_NAME}" ]]; then
    echo "Info: DOCKER_IMAGE_NAME env variable not set. Setting to hawken50"
    export DOCKER_IMAGE_NAME="hawken50"
fi

# Check that the docker image exists, else build it
if [ -z "$(docker images -q "$DOCKER_IMAGE_NAME" 2> /dev/null)" ]; then
    echo "Info: docker image $DOCKER_IMAGE_NAME doesn't exist. Building..."
    docker build -f Dockerfile -t "$DOCKER_IMAGE_NAME" . || \
    (echo "Error: could not build docker image $DOCKER_IMAGE_NAME" && exit 1)
fi

python3 CVPR24_time_eval.py -n "$DOCKER_IMAGE_NAME" -i ./test_demo/imgs \
    -o ./test_demo/runs/"$timestamp"/segs \
    --timing_save_path "$metrics_save_path"
python3 evaluation/compute_metrics.py -s ./test_demo/runs/"$timestamp"/segs \
    -g test_demo/gts -csv_dir "$metrics_save_path"/metrics.csv