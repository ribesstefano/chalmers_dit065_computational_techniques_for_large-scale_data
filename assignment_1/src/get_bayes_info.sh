#!/bin/bash
OUT=serverinfo_bayes.out
ERR=serverinfo_bayes.error
> $OUT
> $ERR
echo "================================================================================" 2>> $ERR 1>> $OUT
echo "Number of CPU cores" 2>> $ERR 1>> $OUT
echo "================================================================================" 2>> $ERR 1>> $OUT
lscpu | egrep 'CPU\(s\)|Core|Socket|Thread' 2>> $ERR 1>> $OUT
echo "================================================================================" 2>> $ERR 1>> $OUT
echo "CPU Type" 2>> $ERR 1>> $OUT
echo "================================================================================" 2>> $ERR 1>> $OUT
lscpu | egrep "Model name| MHz" 2>> $ERR 1>> $OUT
echo "================================================================================" 2>> $ERR 1>> $OUT
echo "Disk Amount" 2>> $ERR 1>> $OUT
echo "================================================================================" 2>> $ERR 1>> $OUT
df -h --total 2>> $ERR 1>> $OUT
echo "================================================================================" 2>> $ERR 1>> $OUT
echo "Login shell virtual memory usage (code+data+stack) in KB" 2>> $ERR 1>> $OUT
echo "================================================================================" 2>> $ERR 1>> $OUT
ps -o vsz= -p "$$" 2>> $ERR 1>> $OUT
# ps auxU ribes | awk '{memory +=$4}; END {print memory }' 2>> $ERR 1>> $OUT
# echo "================================================================================" 2>> $ERR 1>> $OUT
# echo "Login shell memory - detailed information" 2>> $ERR 1>> $OUT
# echo "================================================================================" 2>> $ERR 1>> $OUT
# ps aux | head -1; ps aux | grep ^ribes | more 2>> $ERR 1>> $OUT