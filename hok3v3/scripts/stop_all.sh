#!/bin/bash

ps -e f|grep -E "python|modelpool|eval|train"| grep -v grep | awk '{print $1}'|xargs kill -s 9

