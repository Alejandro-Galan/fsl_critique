#!/bin/bash
# Kills multiple processes only of one experiment


echo "Executing kill script with parameter: ${1}"

pgrep -f "python3 scripts/run_multiple_experiments.py ${1}" | xargs kill -9
sleep 10
ps aux | grep "exp${1}" | grep "python3 scripts/one_run_network.py" | awk '{print $2}' | xargs kill -9
# zps aux | grep "exp${1}" | grep "python3 scripts/one_run_network.py" | awk '{print $2}' | xargs kill -9
# sleep 10
# pgrep -f "python3 scripts/run_multiple_experiments.py ${1}" | xargs kill -9
