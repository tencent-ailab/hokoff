#!/bin/sh

LEVEL_STR=$1
EVAL_NUMBER=$2
CPU_NUMBER=$3
DATASET_VERSION_NAME=$4
BACKEND=$5
DATASET_NAME=$6
ROOT_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))

A0=`echo $LEVEL_STR | awk -F, '{print $1}'`
A1=`echo $LEVEL_STR | awk -F, '{print $2}'`
DATASET_PATH="$ROOT_DIR/datasets/$DATASET_VERSION_NAME/$DATASET_NAME/"

B0=`echo $BACKEND | awk -F, '{print $1}'`
B1=`echo $BACKEND | awk -F, '{print $2}'`
if [ "$B1" = "" ]; then
    B1=$B0
fi
MODEL_PATH="$ROOT_DIR/baselines/$B0/level-$A0/code/actor/model/init,$ROOT_DIR/baselines/$B1/level-$A1/code/actor/model/init"
if [[ "$DATASET_NAME" =~ "gain_gold" ]]; then
    MODEL_PATH="$ROOT_DIR/baselines/gain_gold/$LEVEL_STR/code/actor/model/init,$ROOT_DIR/baselines/gain_gold/$LEVEL_STR/code/actor/model/init"
fi
echo $MODEL_PATH

gc_server_addr="localhost:23432"
if [ "$GAMECORE_SERVER_ADDR" != "" ]
then
    gc_server_addr=$GAMECORE_SERVER_ADDR
fi
LOG_DIR=$ROOT_DIR/offline_sample/logs/$DATASET_NAME
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
# SCRIPT_DIR=$(realpath $(dirname $0))
# cd $SCRIPT_DIR/../


if [ -f "*.log" ]; then
    rm *.log
fi

echo "monitor_server_addr: $monitor_server_addr"

# MAX_EPISODE=${MAX_EPISODE-"-1"}
let actor_num=$CPU_NUMBER-1
cd $ROOT_DIR
echo $ROOT_DIR
while [ "1" == "1" ]
do

  for i in $(seq 0 $actor_num); do
        let actor_id=$i
        echo "actor_id: $actor_id"
        while [ "1" == "1" ]
            do
            actor_cnt=`ps -elf | grep "python offline_sample/sample_entry.py --actor_id=$actor_id " | grep -v grep | wc -l`
            echo "[`date`] actor_id:$actor_id actor_cnt:$actor_cnt"
            if [ $actor_cnt -lt 1 ]; then
                # actor_log=/dev/null
                # if [ -n "$KAIWU_DEV" ]; then
                actor_log=$LOG_DIR/actor_$i.log
                    # rm $actor_log
                # fi
                break
            else actor_id=$(($actor_id+10000))
            fi
        done
  
        echo "[`date`] restart actor_id:$actor_id"
        nohup python offline_sample/sample_entry.py --actor_id=$actor_id \
                                --gc_server_addr=$gc_server_addr \
                                --ai_server_ip=${AI_SERVER_IP-`hostname -I | awk '{print $1;}'`} \
                                --thread_num=1 \
                                --dataset_path="${DATASET_PATH}$i.hdf5" \
                                --agent_models=${MODEL_PATH} \
                                --eval_number=${EVAL_NUMBER} \
                                --backend=$BACKEND \
                                --dataset_name=$DATASET_NAME \
                                >> $actor_log 2>&1 &
        sleep 1
    #   fi
  done; # for

#   if [ -n "$KAIWU_DEV" ]; then
  break
#   fi

  sleep 30

done; # while
