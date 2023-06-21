if [ ! -n "$1" ] 
then
    LEVEL_STR="8,0"
else
    LEVEL_STR=$1
fi
echo $LEVEL_STR $LEVEL_STR

echo "Total actor num:$LEVEL_STR" $2