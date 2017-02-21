#!/usr/bin/env bash

set -eu
set -o pipefail

export PREFIX=${PREFIX:-"/work/env"}

if [[ ! -z $(which tckgen) ]]
then
    echo "[40-mrtrix.sh] found tckgen, not rebuilding."
    exit 0
else
    echo "[40-mrtrix.sh] building mrtrix3."
fi

if [[ ! -d mrtrix3 ]]
then
    git clone https://github.com/mrtrix3/mrtrix3
fi

# g++ / eigen uses a lot of memory for compiling
memtotal=$(grep MemTotal /proc/meminfo | sed 's,\s\+,-,g' | cut -d'-' -f2)
numprocs=$(($memtotal / (4*1024*1024) + 1))

pushd mrtrix3
    git checkout 1755865
    ./configure -nogui
    NUMBER_OF_PROCESSORS=$numprocs ./build
popd
