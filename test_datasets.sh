#!/bin/bash

# Directory containing the files
DIR="/mnt/nas/share-all/kecen/code/Uranus/config/dataset/retrieval"

# Iterate over all files in the directory
for FILE in "$DIR"/*
do
  if [ -f "$FILE" ]; then
    BASENAME=$(basename "$FILE")
    if [[ "$BASENAME" == *car* || "$BASENAME" == *merge* ]]; then
      echo "Skipping $FILE"
      continue
    fi
    echo "Processing $FILE"
    python3 eval.py --model CricaVPR --resume=/mnt/nas/share-all/kecen/data/CricaVPR/CricaVPR.pth --eval_dataset_cfg_file_path "$FILE" --pca_dim 1024 
  fi
done

###############################################################################

# #COSPLACE
# DIR="/mnt/nas/share-all/kecen/code/Uranus/config/dataset/retrieval_"
# Iterate over all files in the directory
# for FILE in "$DIR"/*
# do
#   if [ -f "$FILE" ]; then
#     BASENAME=$(basename "$FILE")
#     if [[ "$BASENAME" == *car* || "$BASENAME" == *merge* ]]; then
#       echo "Skipping $FILE"
#       continue
#     fi
#     echo "Processing $FILE"
#     python3 eval.py --model Cosplace --eval_dataset_cfg_file_path "$FILE"
#   fi
# done