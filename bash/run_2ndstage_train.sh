#/bin/bash
#PBS -N zeyut
#PBS -l select=1:ncpus=20:mem=250gb:ngpus=1:gpu_model=v100:interconnect=any
#PBS -l walltime=48:00:00
#PBS -e /home/zeyut/eat_detection/workspace/job_output
#PBS -o /home/zeyut/eat_detection/workspace/job_output
#PBS -j oe
#PBS -J 1-6


module load cuda/10.2.89-gcc/8.3.1
module load cudnn/7.6.5.32-10.2-linux-x64-gcc/8.3.1-cuda10_2
module load anaconda3/2019.10-gcc/8.3.1
module load ffmpeg/4.2.2-gcc/8.3.1 
source activate torch-1.8

cd /scratch1/zeyut/eat_detection

cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/src/train_2nd_model.py /scratch1/zeyut/eat_detection
cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/src/utils.py /scratch1/zeyut/eat_detection
cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/src/constants.py /scratch1/zeyut/eat_detection
cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/bash/inputs_2nd_train.txt /scratch1/zeyut/eat_detection

inputs=( $(sed -n ${PBS_ARRAY_INDEX}p inputs_2nd_train.txt) )

python ./train_2nd_model.py ${inputs[0]} ${inputs[1]} ${inputs[2]}



