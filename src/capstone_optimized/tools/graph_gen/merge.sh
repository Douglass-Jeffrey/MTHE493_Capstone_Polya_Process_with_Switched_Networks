#!/bin/bash

for i in {0..109}
do
   padded=$(printf "%04d" $i)

   echo "Processing $padded"

   # Merge bins
   python src/capstone_optimized/tools/hist_merge.py \
   results/data_2026-03-31/lower/hist_data_run_${padded}.csv \
   results/data_2026-03-31/lower/merged/hist_data_run_${padded}.csv \
   --k 2

   # Generate heatmap
   python src/capstone_optimized/tools/plot_hist_heatmap.py \
   results/data_2026-03-31/lower/merged/hist_data_run_${padded}.csv

done