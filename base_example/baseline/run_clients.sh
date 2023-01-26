#! /bin/sh

# nohup ls $1 &
# echo $!

nohup python multi_clients.py $1 > localhost.log 2>&1 &
echo $!

nohup ssh mksit@172.16.101.9 -f "cd ~/reverb_demo/base_example/baseline/; ~/anaconda3/envs/reverb/bin/python multi_clients.py $1" > dgx08.log 2>&1 &
echo $!