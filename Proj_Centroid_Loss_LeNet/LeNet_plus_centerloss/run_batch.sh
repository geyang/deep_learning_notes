# rm nohup.out
# sleep 1.0
# echo "nohup log has been removed"
nohup sh train.sh &
sleep 5.0
tail -n 10 -f nohup.out