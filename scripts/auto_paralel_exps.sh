#!/bin/bash
# Expects experiment number as input

if [ "$#" -ne 2 ] && [ "$#" -ne 3 ]; then
    echo "PLEASE PROVIDE EXPERIMENT NUMBER AND NUMBER OF RUNS, OPTIONAL KILLING\n"
    echo "Use: $0 <expNum> <numRuns> [no_kill]"
    exit 1
fi

no_kill=0

if [ "$#" -eq 3 ]; then
    echo "Three arguments provided. Handling extra parameter of not killing"
    no_kill=1
fi

NUM_RUNS=$2 #10 #10
SCRIPT="python3 scripts/run_multiple_experiments.py $1"


if [ $no_kill -eq 0 ]; then
    # Kill possible previous ghost processes
    ./scripts/paralel_executions/kill_p.sh $1
    # Kill concurrency on this experiment
    rm -r my_utils/constants/exp$1
fi

for ((i=1; i<=NUM_RUNS; i++))
do
    echo "Executing $NUM_RUNS experiment $1 iteration $i..."
    $SCRIPT &
    sleep 5
    # ./scripts/paralel_executions/kill_p.sh ${1}
done


wait
# ./scripts/paralel_executions/kill_p.sh $1
echo "All experiments $1 executed."
