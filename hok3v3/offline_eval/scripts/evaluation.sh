#!/bin/sh


LEVEL_STR=$1

EVAL_NUMBER=$2

CPU_NUMBER=$3

MODEL_PATH=$4

TRAIN_STEP=$5

RUN_PREFIX=$6

EXP_ID=$7

OFFLINE_LOG_PATH=$8

TENSORFLOW_OPPO=$9

DATASET_NAME=${10}
ROOT_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))
gc_server_addr="localhost:23432"
if [ "$GAMECORE_SERVER_ADDR" != "" ]
then
    gc_server_addr=$GAMECORE_SERVER_ADDR
fi
LOG_DIR=$ROOT_DIR/offline_eval/logs/$RUN_PREFIX/$TRAIN_STEP
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
unset USE_ZMQ_CURVE
let actor_num=$CPU_NUMBER-1
cd $ROOT_DIR
while [ "1" == "1" ]
do

  for i in $(seq 0 $actor_num); do
        ((actor_id=$i + $EXP_ID * 100))
        while [ "1" == "1" ]
            do
            actor_cnt=`ps -elf | grep "python offline_eval/entry.py --actor_id=$actor_id " | grep -v grep | wc -l`
            echo "[`date`] actor_id:$actor_id actor_cnt:$actor_cnt"
            if [ $actor_cnt -lt 1 ]; then
                actor_log=/dev/null
                if [ -n "$KAIWU_DEV" ]; then
                    actor_log=$LOG_DIR/actor_$i.log
                    # rm $actor_log
                fi
                break
            else actor_id=$(($actor_id+10000))
            fi
        done
  
        echo "[`date`] restart actor_id:$actor_id"
        nohup python offline_eval/entry.py --actor_id=$actor_id \
                            --i=$i \
                              --gc_server_addr=$gc_server_addr \
                              --ai_server_ip=${AI_SERVER_IP-`hostname -I | awk '{print $1;}'`} \
                              --thread_num=1 \
                              --agent_models=${MODEL_PATH} \
                              --eval_number=${EVAL_NUMBER} \
                              --run_prefix=${RUN_PREFIX}\
                              --train_step=${TRAIN_STEP}\
                              --levels=${LEVEL_STR} \
                              --offline_log_path=${OFFLINE_LOG_PATH}\
                              --tensorflow_oppo=${TENSORFLOW_OPPO} \
                              --dataset_name=${DATASET_NAME} \
                              >> $actor_log 2>&1 &
                            #   --mem_pool_addr=$mem_pool_addr \
                            #   --model_pool_addr="localhost:10016" \
                            #   --monitor_server_addr=${monitor_server_addr} \
        sleep 1
    #   fi
  done; # for

  if [ -n "$KAIWU_DEV" ]; then
      break
  fi

  sleep 30

done; # while
