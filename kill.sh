#!/bin/bash

for ((i = 6402447; i < 6402436; i += 1));
do
  scancel --export=JB=$i
  sleep 3
done