#!/bin/sh

# set env variable
# if [ "$GAMECORE_SERVER_ADDR" == "" ]
# then
#     export GAMECORE_SERVER_ADDR="127.0.0.1:23432"
# fi
# echo "[`date`] set env GAMECORE_SERVER_ADDR:$GAMECORE_SERVER_ADDR"


Max_test_time=72000
if [ $max_test_time ];then
    Max_test_time=$max_test_time
fi



LEVEL_STR=$1
EVAL_NUMBER=$2
CPU_NUMBER=$3
DATASET_VERSION_NAME=$4
DATASET_NAME=$5

ROOT_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))


A0=`echo $LEVEL_STR | awk -F, '{print $1}'`
A1=`echo $LEVEL_STR | awk -F, '{print $2}'`
DATASET_PATH="/datasets/$DATASET_VERSION_NAME/$DATASET_NAME/"
MODEL_PATH="$ROOT_DIR/baselines/tensorflow/level-$A0/algorithms/checkpoint,$ROOT_DIR/baselines/tensorflow/level-$A1/algorithms/checkpoint"
echo ${MODEL_PATH}

LOG_DIR=$ROOT_DIR/offline_sample/logs/$DATASET_NAME
rm -rf $LOG_DIR
mkdir -p $LOG_DIR

if [ -f "*.log" ]; then
    rm *.log
fi

echo "monitor_server_addr: $monitor_server_addr"

MAX_EPISODE=${MAX_EPISODE-"-1"}
let actor_num=$CPU_NUMBER-1
cd $ROOT_DIR/offline_sample
while [ "1" == "1" ]
do

  for i in $(seq 0 $actor_num); do
        let actor_id=$i
        echo "actor_id: $actor_id"
        while [ "1" == "1" ]
            do
            actor_cnt=`ps -elf | grep "python entry.py --actor_id=$actor_id " | grep -v grep | wc -l`
            echo "[`date`] actor_id:$actor_id actor_cnt:$actor_cnt"
            if [ $actor_cnt -lt 1 ]; then
                actor_log=/dev/null
                actor_log=$LOG_DIR/actor_$i.log
                # rm $actor_log
                break
            else actor_id=$(($actor_id+10000))
            fi
        done
  
        echo "[`date`] restart actor_id:$actor_id"
        nohup python entry.py --actor_id=$actor_id \
                                --i=$i \
                                --thread_num=1 \
                                --dataset_path="${DATASET_PATH}$i.hdf5" \
                                --agent_models=${MODEL_PATH} \
                                --eval_number=${EVAL_NUMBER} \
                                --dataset_name=$DATASET_NAME \
                                --game_log_path=$LOG_DIR/game_log \
                                >> $actor_log 2>&1 &
                            #   --mem_pool_addr=$mem_pool_addr \
                            #   --model_pool_addr="localhost:10016" \
                            #   --monitor_server_addr=${monitor_server_addr} \
        sleep 1
    #   fi
  done; # for


  break

  sleep 30

done; # while
