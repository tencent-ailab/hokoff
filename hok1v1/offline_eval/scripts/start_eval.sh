#!/bin/bash

# export INTERFACE_SO_NOT_USE_CURVE=1

ROOT_DIR=$(dirname $(dirname $(dirname $(readlink -f $0))))
SCRIPT_DIR=$ROOT_DIR/offline_eval

USE_GPU=true
if [ $use_gpu ];then
    USE_GPU=$use_gpu
fi


if [ ! -n "$1" ] 
then
    LEVEL_STR="8,0"
else
    LEVEL_STR=$1
fi
echo "LEVLES:$LEVEL_STR"

if [ ! -n "$2" ] 
then
    EVAL_NUMBER=1
else
    EVAL_NUMBER=$2
fi
echo "EVAL_NUMBER:$EVAL_NUMBER"
if [ ! -n "$3" ] 
then
    CPU_NUMBER=1
else
    CPU_NUMBER=$3
fi
echo "CPU_NUMBER:$CPU_NUMBER"

if [ ! -n "$4" ] 
then
    TRAIN_STEP=0
else
    TRAIN_STEP=$4
fi
echo "TRAIN_STEP:$TRAIN_STEP"
if [ ! -n "$5" ] 
then
    RUN_PREFIX=0
else
    RUN_PREFIX=$5
fi
echo "RUN_PREFIX:$RUN_PREFIX"


if [ ! -n "$6" ] 
then
    EXP_ID=0
else
    EXP_ID=$6
fi
echo "EXP_ID:$EXP_ID"
if [ ! -n "$7" ] 
then
    OFFLINE_LOG_PATH=0
else
    OFFLINE_LOG_PATH=$7
fi
echo "OFFLINE_LOG_PATH:$OFFLINE_LOG_PATH"
if [ ! -n "$8" ] 
then
    TENSORFLOW_OPPO=0
else
    TENSORFLOW_OPPO=$8
fi
echo "TENSORFLOW_OPPO:$TENSORFLOW_OPPO"
if [ ! -n "$9" ] 
then
    DATASET_NAME='level-0-0'
else
    DATASET_NAME=$9
fi
echo "DATASET_NAME:$DATASET_NAME"

Max_test_time=$(($EVAL_NUMBER*420))
if [ $max_test_time ];then
    Max_test_time=$max_test_time
fi

MODEL_PATH="$OFFLINE_LOG_PATH/${RUN_PREFIX}/${TRAIN_STEP}_model"

LOG_DIR=$ROOT_DIR/offline_eval/logs/$RUN_PREFIX/$TRAIN_STEP
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
rm -rf $ROOT_DIR/log/info*

cd $ROOT_DIR/offline_eval/; bash scripts/evaluation.sh $LEVEL_STR $EVAL_NUMBER $CPU_NUMBER $MODEL_PATH $TRAIN_STEP $RUN_PREFIX $EXP_ID $OFFLINE_LOG_PATH $TENSORFLOW_OPPO $DATASET_NAME

start_time=` date  +%s`
# Code testing
cd $ROOT_DIR;
echo "Total actor num:$CPU_NUMBER"
cnt=$(($CPU_NUMBER-1))
while [ $(( $(date +%s) - start_time )) -lt $Max_test_time ]
do  
    done_cpu=0
    
    for i in $(seq 0 $cnt); do
        echo '!!!!'
        echo grep -c "close ip" $LOG_DIR/actor_$i.log
        if [ $(grep -c "close ip" "$LOG_DIR/actor_$i.log") = 2 ]
        then
            done_cpu=$(($done_cpu+1))
        fi
    done;
    echo "Current done actor num: $done_cpu"
    if [ "$done_cpu" = "$CPU_NUMBER" ]
    then
        echo "All actors are done."
        break
    else
        sleep 6s
    fi
done
