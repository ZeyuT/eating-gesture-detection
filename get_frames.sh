#/bin/bash
#PBS -N zeyut
#PBS -l select=1:ncpus=8:mem=100gb:interconnect=any
#PBS -l walltime=72:00:00
#PBS -e /home/zeyut/eat_detection/job_output
#PBS -o /home/zeyut/eat_detection/job_output
#PBS -j oe

module load anaconda3/5.0.1-gcc/8.3.1
module load ffmpeg/4.2.2-gcc/8.3.1 
source activate tf-2.2
cd /scratch1/zeyut/eat_detection

cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/get_frames.py /scratch1/zeyut/eat_detection

python get_frames.py 8


