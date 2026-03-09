#!/bin/bash

#SBATCH -J lindsay
#SBATCH -p general
#SBATCH -o lindsay_%j.txt
#SBATCH -e lindsay_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH -A r00213

mkdir ./Evolutions_2N/$JB;
cd ./Evolutions_2N/$JB;
time ../../main.exe $JB;
cd ../../;
