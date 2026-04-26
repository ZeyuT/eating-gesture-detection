#/bin/bash
#PBS -N res_lstm
#PBS -l select=1:ncpus=16:mem=150gb:ngpus=1:gpu_model=v100:interconnect=any
#PBS -l walltime=48:00:00
#PBS -e /home/zeyut/meta/workspace/job_output
#PBS -o /home/zeyut/meta/workspace/job_output
#PBS -j oe
#PBS -J 1-5

module load anaconda3/2022.05-gcc/9.5.0
module load ffmpeg/4.4.1-gcc/9.5.0
source activate torch-1.13

cd /home/zeyut/meta/workspace/eating-gesture-detection/src

inputs=( $(sed -n ${PBS_ARRAY_INDEX}p ../bash/EatSense/inputs_multi_train4.txt) )

python ./EatSense/train_model.py ${inputs[0]} ${inputs[1]} ${inputs[2]} ${inputs[3]} ${inputs[4]} ${inputs[5]} ${inputs[6]}

