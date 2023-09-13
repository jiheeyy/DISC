#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=logs/array_%A.out
#SBATCH --error=logs/array_%A.err
#SBATCH --time=4:00:00
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jiheeyou@rcc.uchicago.edu  # mail notification for the job

mkdir -p logs
source activate disc
cd $HOME/disc/shap
python shapviz.py