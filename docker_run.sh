#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
GPUs=$1
IFS=',' read -ra GPU_ARRAY <<< "$GPUs"
name="${USER}_dexter_GPU_${GPUs//,/_}_${HASH}"

echo "Launching container named '${name}' on GPUs '${GPUs}'"

cmd=docker

echo "${@:2}" 2>&1 >> log.txt

docker run -it \
    --name "$name" \
    -u 0 \
    -v "$(pwd)":/home/duser/entryfolder \
    -e PYTHONPATH=/home/duser/entryfolder \
    -t dexter \
    /bin/bash -c "${@:2}"