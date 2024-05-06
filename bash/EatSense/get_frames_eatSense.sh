#/bin/bash
#PBS -N zeyut
#PBS -l select=1:ncpus=20:mem=300gb:interconnect=any
#PBS -l walltime=12:00:00
#PBS -e /home/zeyut/meta/workspace/job_output
#PBS -o /home/zeyut/meta/workspace/job_output
#PBS -j oe

module load anaconda3/2022.05-gcc/9.5.0
module load ffmpeg/4.4.1-gcc/9.5.0

source activate torch-1.13

cd /home/zeyut/meta/workspace/eating-gesture-detection/src

# python ./EatSense/extract_frames.py 8
python ./EatSense/process_frames.py 8

# python ./EatSense/extract_frames.py 5
python ./EatSense/process_frames.py 5
