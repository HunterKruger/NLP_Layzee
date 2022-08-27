#!/bin/bash

source /opt/app/conda/etc/profile.d/conda.sh
conda activate /home/powerop/work/conda/envs/nlp
conda info --envs

cur_dateTime="`date +%Y-%m-%d_%H-%M-%S`"  
echo $cur_dateTime

root_path="ABT_S_0822"
echo $root_path

sed -i "1s/.*/FILE_NAME = '$root_path'/" config.py

if [ ! -d ../experiments/${root_path}  ];then
  mkdir ../experiments/${root_path}
else
  rm -rf ../experiments/${root_path}
  mkdir ../experiments/${root_path}
fi

export LANG="en_US.UTF-8" 
nohup python -u train_val_test.py >../experiments/${root_path}/log.txt 2 >&1 &
