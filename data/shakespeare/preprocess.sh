#!/usr/bin/env bash

# download data and convert to .json format

RAWTAG=""
if [[ $@ = *"--raw"* ]]; then
  RAWTAG="--raw"
fi
if [ ! -d "data/all_data" ] || [ ! "$(ls -A data/all_data)" ]; then
    echo "first download and unzip the data directory"
    exit 0
fi

NAME="shakespeare"

cd ../utils

bash ./preprocess.sh --name $NAME $@

cd ../$NAME
