#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Array of model sizes to process
models=("yolo11n" "yolo11s" "yolo11m")

# Loop through each model size
for model in "${models[@]}"; do
    echo "Processing model: ${model}"
    
    # Strip .pt extension if present to get the base name
    base_name="${model%.pt}"
    
    # Get just the filename without path for directory naming
    dir_name=$(basename "${base_name}")
    
    # Create folders for each precision
    mkdir -p "./${dir_name}/FP32/" "./${dir_name}/FP16/"
    
    # Export FP32 model
    echo "Exporting ${model} in FP32 precision..."
    yolo export model="${model}" format=openvino half=false imgsz=640,640
    mv "${base_name}_openvino_model/${base_name}."* "./${dir_name}/FP32/"
    rm -rf "${base_name}_openvino_model"

    # Export FP16 model
    echo "Exporting ${model} in FP16 precision..."
    yolo export model="${model}" format=openvino half=true imgsz=640,640
    mv "${base_name}_openvino_model/${base_name}."* "./${dir_name}/FP16/"
    rm -rf "${base_name}_openvino_model"

done

echo "All models processed successfully!"
