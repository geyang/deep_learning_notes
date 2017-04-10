#!/usr/bin/env bash
nohup sh ./start_visdom.sh &
echo "visdom server is running"

nohup sh ./jupyter_5021.sh &
watch tail nohup.out
