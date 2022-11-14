#!/usr/bin/env bash
# run from inside scripts

for (( i = 1; i<= 100; i=$(( $i+4 ))))
do
  echo "$i - $(($i+4))"
  python generate_from_pompy.py -p "../animations/$i" &
  python generate_from_pompy.py -p "../animations/$(($i+1))" &
  python generate_from_pompy.py -p "../animations/$(($i+2))" &
  python generate_from_pompy.py -p "../animations/$(($i+3))" &
  wait
done
