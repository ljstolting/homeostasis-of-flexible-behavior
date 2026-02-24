#!/bin/bash

for ((i = 0; i < 100; i += 1));
do
  sbatch --export=JB=$i lindsaygeneral1.sh
  sleep 3
done
