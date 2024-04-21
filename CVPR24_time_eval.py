"""
The code was adapted from the MICCAI FLARE Challenge
https://github.com/JunMa11/FLARE

The testing images will be evaluated one by one.

Folder structure:
CVPR24_time_eval.py
- team_docker
    - teamname.tar.gz # submitted docker containers from participants
- test_demo
    - imgs
        - case1.npz  # testing image
        - case2.npz  
        - ...   
- demo_seg  # segmentation results
    - case1.npz  # segmentation file name is the same as the testing image name
    - case2.npz  
    - ...
"""

import os
join = os.path.join
import shutil
import time
import torch
import argparse
from collections import OrderedDict
import pandas as pd

parser = argparse.ArgumentParser('Segmentation efficiency eavluation for docker containers', add_help=False)
parser.add_argument('-i', '--test_img_path', default='./test_demo/imgs', type=str, help='testing data path')
parser.add_argument('-o','--segs_save_path', default='./test_demo/segs', type=str, help='segmentation output path')
parser.add_argument('-t','--timing_save_path', default='./test_demo/', type=str, help='running time data output path')
parser.add_argument('-n','--docker_image_name', type=str, help='File name of the docker image')
args = parser.parse_args()

test_img_path = args.test_img_path
segs_save_path = args.segs_save_path
timing_save_path = args.timing_save_path
docker_image_name = args.docker_image_name

os.makedirs(segs_save_path, exist_ok=True)
input_temp = './inputs_temp/'
output_temp = './outputs_temp/'

test_cases = sorted(os.listdir(test_img_path))

try:
    # create temp folers for inference one-by-one
    if os.path.exists(input_temp):
        shutil.rmtree(input_temp)
    if os.path.exists(output_temp):
        shutil.rmtree(output_temp)
    os.makedirs(input_temp)
    os.makedirs(output_temp)
    # load docker and create a new folder to save segmentation results
    teamname = docker_image_name.split('.')[0].lower()
    print('Team name: ', teamname)
    # os.system('docker image load -i {}'.format(docker_image_name))
    team_outpath = segs_save_path
    if os.path.exists(team_outpath):
        shutil.rmtree(team_outpath)
    os.mkdir(team_outpath)
    os.system('chmod -R 777 {} {}'.format(input_temp, output_temp))
    metric = OrderedDict()
    metric['CaseName'] = []
    metric['InferenceTime'] = []
    metric['RunningTime'] = []
    # To obtain the running time for each case, testing cases are inferred one-by-one
    for case in test_cases:
        shutil.copy(join(test_img_path, case), input_temp)
        cmd = 'docker container run -m 8G --name {} --rm -v {}:/workspace/inputs/ -v {}:/workspace/outputs/ {}:latest /bin/bash -c "sh predict.sh" '.format(teamname, input_temp, output_temp, teamname)
        print(teamname, ' docker command:', cmd, '\n', 'testing image name:', case)
        start_time = time.time()
        os.system(cmd)
        real_running_time = time.time() - start_time
        print(f"{case} finished! Inference time: {real_running_time}")
        # save metrics
        efficiency_df = pd.read_csv(join(output_temp, "efficiency.csv"))
        metric['CaseName'].append(case)
        metric['InferenceTime'].append(efficiency_df.iloc[0]['time'])
        metric['RunningTime'].append(real_running_time)
        os.remove(join(input_temp, case))  
        seg_name = case
        try:
            os.rename(join(output_temp, seg_name), join(team_outpath, seg_name))
        except:
            print(f"{join(output_temp, seg_name)}, {join(team_outpath, seg_name)}")
            print("Wrong segmentation name!!! It should be the same as image_name")
    metric_df = pd.DataFrame(metric)
    running_time_path = join(timing_save_path, 'running_time.csv')
    print("Running time data saved to:", running_time_path)
    metric_df.to_csv(running_time_path, index=False)
    shutil.rmtree(input_temp)
    shutil.rmtree(output_temp)
except Exception as e:
    print(e)