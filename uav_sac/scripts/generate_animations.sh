#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for (( i = 1; i<= $1; i=$(( $i+4 ))))
do
  echo "$i - $(($i+4))"
  python generate_from_pompy.py -p "${SCRIPT_DIR}/../../animations/$i" &
  python generate_from_pompy.py -p "${SCRIPT_DIR}/../../animations/$(($i+1))" &
  python generate_from_pompy.py -p "${SCRIPT_DIR}/../../animations/$(($i+2))" &
  python generate_from_pompy.py -p "${SCRIPT_DIR}/../../animations/$(($i+3))" &
  wait
done
