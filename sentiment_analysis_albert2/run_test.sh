#!/bin/bash

source /opt/app/conda/etc/profile.d/conda.sh
conda activate /home/powerop/work/conda/envs/nlp
conda info --envs

cur_dateTime="`date +%Y-%m-%d_%H-%M-%S`"  
echo $cur_dateTime

root_path="A002"
echo $root_path

sed -i "1s/.*/FILE_NAME = '$root_path'/" config.py

export LANG="en_US.UTF-8" 
nohup python -u test.py >../experiments/${root_path}/test_log.txt 2 >&1 &

