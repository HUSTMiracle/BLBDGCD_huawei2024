#! /bin/bash

#SBATCH --job-name=yolov9_val         
#SBATCH --gres=gpu:1
#SBATCH --error=val/val_%j.log                

conda init bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate yolov9
conda info --envs
python --version
export CUDA_VISIBLE_DEVICES=1
nvidia-smi

python /home/gresham/Yolov9/model/yolo_v9/detect.py \
--img 3072 --conf 0.1 \
--source /home/gresham/Yolov9/input/01_Mouse_bite.bmp \
--weights /home/gresham/Yolov9/model/yolov9.pt \
--save-txt 
