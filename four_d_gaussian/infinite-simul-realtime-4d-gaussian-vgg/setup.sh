#!/bin/bash

# Parse command line arguments
INSTALL=false
DOWNLOAD=false
GAUSSIAN=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --install)
      INSTALL=true
      shift
      ;;
    --download)
      DOWNLOAD=true
      shift
      ;;
    --gaussian)
      GAUSSIAN=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Install requirements if --install flag is set
if [ "$INSTALL" = true ]; then
    sudo apt update
    apt update && apt install -y cuda-toolkit-11-7

    chmod +x ./third_party/infinite-simul-spacetime-gaussian/script/setup.sh

    cd third_party/infinite-simul-spacetime-gaussian
    bash ./script/setup.sh

    cd ../../

    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# If --download flag is set, run the dataset download script
if [ "$DOWNLOAD" = true ]; then
    echo "Downloading datasets..."
    chmod +x datasets/download.sh
    ./datasets/download.sh

    export QT_QPA_PLATFORM=offscreen


    # Run the preprocessor
    echo "Running dataset preprocessor..."
    xvfb-run python3 datasets/preprocessor.py
fi

# If --gaussian flag is set, run the training script
if [ "$GAUSSIAN" = true ]; then
    echo "Running Gaussian training..."
    python3 datasets/train_gaussian.py
fi

