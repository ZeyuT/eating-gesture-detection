#/bin/bash
#PBS -N zeyut
#PBS -l select=1:ncpus=16:mem=150gb:ngpus=2:gpu_model=v100:interconnect=any
#PBS -l walltime=72:00:00
#PBS -e /home/zeyut/eat_detection/workspace/job_output
#PBS -o /home/zeyut/eat_detection/workspace/job_output
#PBS -j oe
#PBS -J 1-3

module load cuda/11.4.1-gcc/9.3.0
module load cudnn/8.0.4.30-11.1-linux-x64-gcc/8.4.1
module load anaconda3/2019.10-gcc/8.3.1
module load ffmpeg/4.2.2-gcc/8.3.1 
source activate torch-1.8

cd /scratch1/zeyut/eat_detection/reimplementation

cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/reimplementation/*.py /scratch1/zeyut/eat_detection/reimplementation/
cp -r /home/zeyut/eat_detection/workspace/eating-gesture-detection/reimplementation/models/ /scratch1/zeyut/eat_detection/reimplementation/
cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/bash/inputs_reimplementation2.txt /scratch1/zeyut/eat_detection/reimplementation/

inputs=( $(sed -n ${PBS_ARRAY_INDEX}p inputs_reimplementation2.txt) )

python ./train_model.py ${inputs[0]} ${inputs[1]} ${inputs[2]} ${inputs[3]}

