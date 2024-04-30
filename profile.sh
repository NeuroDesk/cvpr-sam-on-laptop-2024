#!/bin/bash

# Usage:
# Parameter 1: File name of the test case (Not full path)
#               The file will be taken from test_demo/imgs
# Parameter 2: Path to the root of the repo. If not given,
#               will assume that the cwd is the root
# 
# Prerequisites: The test data and model should be downloaded
# and placed in the correct folders according to the repo README

test_file_name="$1"
repo_root="$2"
timestamp=$(date +%d-%m-%Y-%H-%M-%S)

# if the repo root path is not given, assume the current path is the repo root
if [[ -n "$2" ]]; then
    cd "${repo_root}"
else
    repo_root=$(pwd)
fi

if [[ -z "$1" ]]; then
    test_file_name=3DBox_CT_demo.npz
fi

mkdir -p workspace
cp -R -u -p "${repo_root}"/test_demo/* workspace
mkdir workspace/"$timestamp"


python3 -m kernprof -l "${repo_root}/CVPR24_LiteMedSamOnnx_infer_profile.py" \
    -i ./workspace/imgs -o ./workspace/"$timestamp"/outputs \
    -f "./workspace/imgs/$test_file_name" -model_path work_dir/LiteMedSAM
python3 -m line_profiler -rmt "CVPR24_LiteMedSamOnnx_infer_profile.py.lprof" > "profiler_output_$test_file_name.txt"

echo "Inspected profiler output has been written to profiler_output_$test_file_name.txt"