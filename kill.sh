#!/bin/bash

for ((i = 5851782; i < 5851854; i += 1));
do
  scancel --export=JB=$i
  sleep 3
done