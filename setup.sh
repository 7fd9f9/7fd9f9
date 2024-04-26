#!/bin/bash

wget https://github.com/data61/MP-SPDZ/releases/download/v0.3.8/mp-spdz-0.3.8.tar.xz
tar -xf mp-spdz-0.3.8.tar.xz
rm mp-spdz-0.3.8.tar.xz
cp -r src/. mp-spdz-0.3.8/Programs/Source

cd mp-spdz-0.3.8 || exit
./Scripts/tldr.sh

./compile.py -R 64 -Z 3 bfs-unit-tests
./compile.py -R 64 -Z 3 max-flow-unit-tests

./Scripts/setup-ssl.sh 3
./Scripts/ring.sh bfs-unit-tests
./Scripts/ring.sh max-flow-unit-tests
./Scripts/ps-rep-ring.sh bfs-unit-tests
./Scripts/ps-rep-ring.sh max-flow-unit-tests