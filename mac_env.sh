# Activate the conda environment
CONDA_SUBDIR=osx-64
conda create -n bio python=3.6.8
eval "$(conda shell.bash hook)"
source activate bio

# Set the conda subdirectory and update the environment
conda config --env --set subdir osx-64
conda env update --file environment_mac.yml --prune

# Install PyTorch
pip install https://download.pytorch.org/whl/cpu/torch-1.4.0-cp36-none-macosx_10_7_x86_64.whl

# Install the IPython kernel for Jupyter
python -m ipykernel install --user --name=bio
