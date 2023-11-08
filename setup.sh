#!/usr/bin/env bash

# Load environment variables
set -a
source .env
set +a

# Create virtual environment
conda_env="nvflare"
create_conda_env="conda create -n \$conda_env python=3.9.0 --y"
activate_conda_env="conda activate \$conda_env"

{
    eval "$activate_conda_env"
} || {
    echo "Creating new conda environment $conda_env" && 
    eval "$create_conda_env" && 
    eval "$activate_conda_env" &&
    echo "DONE"
}

venv=${1:-venv}

if [ ! -d "$venv" ]; then
  python -m venv $venv
fi

source $venv/bin/activate

# Install packages
pip install --upgrade pip
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch==1.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nvflare nibabel torch-summary matplotlib ipykernel
# pip install libLAS

pip freeze > requirements.txt

if [ "$TASK_NUM" == "1" ]
then
  echo "Choosing Task $TASK_NUM"
  wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
  tar -xvf Task01_BrainTumour.tar
  mkdir -p models/task01
elif [ "$TASK_NUM" == "2" ]
then
  echo "Choosing Task $TASK_NUM"
  wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar
  tar -xvf Task02_Heart.tar
  mkdir -p models/task02
elif [ "$TASK_NUM" == "6" ]
  echo "Choosing Task $TASK_NUM"
  wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar
  tar -xvf Task06_Lung.tar
  mkdir -p models/task06
else
  echo "Task number $TASK_NUM not found!"

# # Download datasets
# wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
# wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar
# wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar

# # Untar dataset
# tar -xvf Task01_BrainTumour.tar
# tar -xvf Task02_Heart.tar
# tar -xvf Task06_Lung.tar

# # Create directories for each model training
# mkdir models
