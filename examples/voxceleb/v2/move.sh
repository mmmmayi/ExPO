#!/bin/bash

# 设置路径A和路径B
path_a="/hpctmp/ma_yi/dataset_vox1/voxceleb1/wav"
path_b="/hpctmp/ma_yi/dataset_vox1/voxceleb1/test/wav"

# 遍历路径A中的所有文件夹
for folder in "$path_a"/*; do
  if [ -d "$folder" ]; then # 确保是一个文件夹
    folder_name=$(basename "$folder")
    # 检查是否在路径B中存在同名文件夹
    if [ -d "$path_b/$folder_name" ]; then
      # 移动并覆盖路径B中的文件夹
      echo "Moving and overwriting $folder_name to $path_b"
      rm -rf "$path_b/$folder_name"
      mv "$folder" "$path_b"
      #rm -rf "$path_a/$folder_name"
    fi
  fi
done