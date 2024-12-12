#!/bin/bash

# export INTERFACE_SO_NOT_USE_CURVE=1

SCRIPT_DIR=$(dirname $(dirname $(readlink -f $0)))

if [ ! -n "$1" ] ### if exists one parameter ###
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
    DATASET_VERSION_NAME='tmpversion1'
else
    DATASET_VERSION_NAME=$4
fi

A0=`echo $LEVEL_STR | awk -F, '{print $1}'`
A1=`echo $LEVEL_STR | awk -F, '{print $2}'`
if [ ! -n "$5" ] 
then
    DATASET_NAME="level-$A0-$A1/"
else
    DATASET_NAME=$5
fi
echo "DATASET_NAME:$DATASET_NAME"
LOG_DIR=$SCRIPT_DIR/logs/$DATASET_NAME
rm -rf $LOG_DIR
mkdir -p $LOG_DIR ### recursive make dirs ###
cd $SCRIPT_DIR/scripts; bash sample.sh $LEVEL_STR $EVAL_NUMBER $CPU_NUMBER $DATASET_VERSION_NAME $DATASET_NAME

echo "start end!!!!"
Max_test_time=$(($EVAL_NUMBER*1200)) ### every game 20 minutes ###
if [ $max_test_time ];then
    Max_test_time=$max_test_time
fi

start_time=` date  +%s`
# Code testing
cd $SCRIPT_DIR/logs;
echo "Total actor num:$CPU_NUMBER"
cnt=$(($CPU_NUMBER-1))
while [ $(( $(date +%s) - start_time )) -lt $Max_test_time ]
do  
    done_cpu=0
    for i in $(seq 0 $cnt); do
        if [ $(grep -c "close ip" "$LOG_DIR/actor_$i.log") = 2 ]
        then
            done_cpu=$(($done_cpu+1))
        fi
    done;
    echo "Current done actor num: $done_cpu, time: $(( $(date +%s) - start_time ))"
    if [ "$done_cpu" == "$CPU_NUMBER" ]
    then
        echo "All actors are done."
        break
    else
        sleep 60s
    fi
done
