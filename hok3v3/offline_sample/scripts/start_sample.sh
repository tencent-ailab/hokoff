#!/bin/bash


unset USE_ZMQ_CURVE

SCRIPT_DIR=$(dirname $(dirname $(readlink -f $0)))

# ln -sf $SCRIPT_DIR/hero_reward.txt /.hok_env/hok/AILab/ai_config/5v5/reward/hero_reward.txt




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
    DATASET_VERSION_NAME='3v3version2'
else
    DATASET_VERSION_NAME=$4
fi
if [ ! -n "$5" ] 
then
    BACKEND='tensorflow'
else
    BACKEND=$5
fi
echo "BACKEND:$BACKEND"
A0=`echo $LEVEL_STR | awk -F, '{print $1}'`
A1=`echo $LEVEL_STR | awk -F, '{print $2}'`
if [ ! -n "$6" ] 
then
    DATASET_NAME="level-$A0-$A1/"
else
    DATASET_NAME=$6
fi
echo "DATASET_NAME:$DATASET_NAME"
LOG_DIR=$SCRIPT_DIR/logs/$DATASET_NAME
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
cd $SCRIPT_DIR/scripts; bash sample.sh $LEVEL_STR $EVAL_NUMBER $CPU_NUMBER $DATASET_VERSION_NAME $BACKEND $DATASET_NAME

echo "start end!!!!"
Max_test_time=$(($EVAL_NUMBER*420))
if [ $max_test_time ];then
    Max_test_time=$max_test_time
fi

start_time=` date  +%s`
# Code testing
# cd $SCRIPT_DIR/logs;
echo "Total actor num:$CPU_NUMBER"
cnt=$(($CPU_NUMBER-1))
echo "cnt: $cnt"
# while [ $[` date  +%s` - $start_time] -lt $Max_test_time ]
while [ $(( $(date +%s) - start_time )) -lt $Max_test_time ]
do  
    done_cpu=0
    for i in $(seq 0 $cnt); do
        # if [ ` grep -c "close ip" $LOG_DIR/actor_$i.log ` == 2 ]
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