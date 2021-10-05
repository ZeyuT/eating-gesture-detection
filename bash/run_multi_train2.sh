#/bin/bash
#PBS -N zeyut
#PBS -l select=1:ncpus=16:mem=150gb:ngpus=1:gpu_model=v100:interconnect=any
#PBS -l walltime=48:00:00
#PBS -e /home/zeyut/eat_detection/workspace/job_output
#PBS -o /home/zeyut/eat_detection/workspace/job_output
#PBS -j oe
#PBS -J 1-10


module load cuda/10.2.89-gcc/8.3.1
module load cudnn/7.6.5.32-10.2-linux-x64-gcc/8.3.1-cuda10_2
module load anaconda3/2019.10-gcc/8.3.1
module load ffmpeg/4.2.2-gcc/8.3.1 
source activate torch-1.8

cd /scratch1/zeyut/eat_detection

cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/train_model.py /scratch1/zeyut/eat_detection
cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/models.py /scratch1/zeyut/eat_detection
cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/utils.py /scratch1/zeyut/eat_detection
cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/constants.py /scratch1/zeyut/eat_detection
cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/inputs_multi_train2.txt /scratch1/zeyut/eat_detection

inputs=( $(sed -n ${PBS_ARRAY_INDEX}p inputs_multi_train2.txt) )

python ./train_model.py ${inputs[0]} ${inputs[1]} ${inputs[2]} ${inputs[3]} ${inputs[4]} ${inputs[5]} ${inputs[6]} ${inputs[7]}


