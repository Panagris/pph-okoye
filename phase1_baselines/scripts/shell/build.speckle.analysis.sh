#!/bin/bash

g++ -O3 -fopenmp speckle_analysis.cpp -o speckle_analysis  `pkg-config --cflags --libs opencv4` && echo "./speckle_analysis <video_path> <num_cores> <target_fps> [kernel_size=5] [exposure_time_ms=3.0]"
