#!/usr/bin/env bash
# run from inside scripts

for (( i = 1; i<= 10; i++))
do
  python generate_from_pompy.py -p "../animations/$i"
done
