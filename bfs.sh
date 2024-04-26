#!/usr/bin/env bash

if [ "$1" = "mal" ]; then
  CMD="./Scripts/ps-rep-ring.sh"
else
  CMD="./Scripts/ring.sh"
fi
PROTOCOL=$2
NUM_NODES=$3
CMD="$CMD --batch-size 1000000 --bits-from-squares run-bfs-$PROTOCOL-$NUM_NODES"
if [ "$4" != "with-overflows" ]; then
  MAX_VAL="9223372036854775807" # 2^63 - 1
  CMD="$CMD-$MAX_VAL"
fi

cd mp-spdz-0.3.8 || exit
./compile.py -R 64 -Z 3 run-bfs "$PROTOCOL" "$NUM_NODES" $MAX_VAL || exit
echo "$CMD" | bash
