#!/bin/bash

cd /home/powerop/work/im_high_freq/code
source /opt/app/conda/etc/profile.d/conda.sh
conda activate /home/powerop/work/conda/envs/high_freq
conda info --envs


#运行出错则退出
set -e

cur_dateTime="`date +%Y-%m-%d`"  
echo $cur_dateTime

kinit -kt /etc/keytab/htl_ai.keytab htl_ai

export PATH=$PATH:/opt/app/spark/bin

current_date=$(date +%Y-%m-%d)
operate_date=$(date +%Y-%m-%d --date='-1 day')
before2_date=$(date +%Y-%m-%d --date='-2 day')
before8_date=$(date +%Y-%m-%d --date='-8 day')

hive -e"
    SELECT * FROM tmp_htl_ai_db.im_weekly
">input_data.txt

# 判读取数成功与否，失败则终止脚本运行
row_num=$(cat input_data.txt | wc -l)
echo ${row_num}
if (( $row_num < 5 )); then
    echo "get data failed"
    exit 1
fi
echo "============================取数成功=============================="


echo "============================运行py脚本=============================="
export LANG="en_US.UTF-8" 
python -u all_process_weekly.py
echo "============================py脚本执行完毕=============================="


echo "============================删除临时文件=============================="
rm input_data.txt
rm bigram.txt
rm trigram.txt
rm tokenized.txt
echo "============================shell全部执行完毕=============================="
