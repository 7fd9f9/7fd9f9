#!/usr/bin/env bash

if [ "$1" = "mal" ]; then
  CMD="./Scripts/ps-rep-ring.sh"
else
  CMD="./Scripts/ring.sh"
fi
PROTOCOL=$2
NUM_NODES=$3
CMD="$CMD --batch-size 1000000 --bits-from-squares run-max-flow-$PROTOCOL-$NUM_NODES"
if [ "$4" != "" ]; then
  BIT_LENGTH=$4
  CMD="$CMD-$BIT_LENGTH"
fi

cd mp-spdz-0.3.8 || exit
./compile.py -R 64 -Z 3 run-max-flow "$PROTOCOL" "$NUM_NODES" $BIT_LENGTH || exit
echo "$CMD" | bash
