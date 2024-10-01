#!/bin/bash
#SBATCH --partition=elec-vca.gpu.q
#SBATCH --nodes=1                               # Specify the amount of A100 Nodes with 4 A100 GPUs (single GPU 128 SBUs/hour, 512 SBUs/hour for an entire node)
#SBATCH --ntasks=1                              # Specify the number of tasks
#SBATCH --cpus-per-task=24                      # Specify the number of CPUs/task
#SBATCH --gpus=titanrtx.24gb:2                 # Specify the number of GPUs to use
#SBATCH --time=120:00:00                        # Specify the maximum time the job can run

export OUTPUT_FOLDER=DiRA_SurgeNet                                                    # Define name of output folder
cd /home/elec001/20172619/SSL/dino/ || return                                              # Navigate to the location of the code (ckusters account)
export WANDB_API_KEY=1cf878a1b1aafcd37a1f6e6ba8fdd18ba1c4affb
export WANDB_DIR=/home/elec001/20172619/SSL/DiRA/experiments/$OUTPUT_FOLDER/wandb
export WANDB_CONFIG_DIR=/home/elec001/20172619/SSL/DiRA/experiments/$OUTPUT_FOLDER/wandb
export WANDB_CACHE_DIR=/home/elec001/20172619/SSL/DiRA/$OUTPUT_FOLDER/wandb
export WANDB_START_METHOD="thread"
wandb login

### SETUP DIRECTORY TO WORK IN ###
cd /vast.mnt/home/20172619/SSL/DiRA || return


### RUN DiRA training on SurgeNet ###
srun apptainer exec --bind /elec003.mnt:/elec003.mnt --bind /vast.mnt:/vast.mnt --nv /elec003.mnt/project/elec-vca-uppergi/Docker/Tim/dira_v5.sif  torchrun --nnodes 1 --nproc_per_node 2 main_DiRA.py \
    '/elec003.mnt/project/elec-vca-uppergi/Datasets/ssl-datasets/SurgeNetXL' \
    -a caformer_s18 \
    --lr=0.03 \
    --batch-size=2048 \
    --dist-url='tcp://localhost:10001' \
    --multiprocessing-distributed \
    --world-size=1 \
    --rank=0 \
    --experimentname='SurgeNet-DiRA' \
    --epochs=50 \
    --output_dir='/vast.mnt/home/20172619/SSL/DiRA/output/' \
