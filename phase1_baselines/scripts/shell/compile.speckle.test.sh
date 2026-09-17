#!/bin/bash

# Function to print usage instructions
usage() {
    echo "Usage: $0 [OPTION]"
    echo "Options:"
    echo "  --no-opencv    Build without OpenCV (-DNO_OPENCV)"
    echo "  --opencv       Build with OpenCV support"
    echo "  --help         Display this help message"
    exit 1
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    usage
fi

# Process the command-line argument
case "$1" in
    --no-opencv)
        echo "Building without OpenCV (-DNO_OPENCV)..."
        g++ -O3 -fopenmp -DNO_OPENCV speckle_test-v2.cpp -o speckle_test
	matched=true
        ;;
    --opencv)
        echo "Building with OpenCV support..."
        g++ -O3 -fopenmp speckle_test-v2.cpp -o speckle_test `pkg-config --cflags --libs opencv4`
	matched=true
        ;;
    --help|-h)
        usage
        ;;
    *)
        echo "Error: Invalid option '$1'"
        usage
        ;;
esac

if [ "$matched" = true ]; then
	echo "Usage: ./speckle_test <video_type> <resolution> [source_fps] [target_fps] [kernel_size] [exposure_time_ms]"
fi
