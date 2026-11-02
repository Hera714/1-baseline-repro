#!/bin/bash

# Install Git LFS
git lfs install

# Clone the Depth-Anything-V2 repository from Hugging Face
git clone https://huggingface.co/spaces/depth-anything/Depth-Anything-V2

# Install required dependencies
pip install -r Depth-Anything-V2/requirements.txt

# Install the 'spaces' package
pip install spaces

echo "Installation completed successfully!"
