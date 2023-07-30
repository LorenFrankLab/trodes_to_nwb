#!/bin/bash

FILENAME="$1"
JOB_ID="${FILENAME%.*}"

now=$(date +"%Y%m%d_%H%M")
outname="out_${JOB_ID}_${now}.log"

echo "***" >> "$outname"

nohup python -u "$FILENAME" >> "$outname" 2>&1 &

