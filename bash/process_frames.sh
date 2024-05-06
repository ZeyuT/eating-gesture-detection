#/bin/bash
#PBS -N zeyut
#PBS -l select=1:ncpus=20:mem=120gb:interconnect=any
#PBS -l walltime=48:00:00
#PBS -e /home/zeyut/eat_detection/workspace/job_output
#PBS -o /home/zeyut/eat_detection/workspace/job_output
#PBS -j oe

module load anaconda3/5.0.1-gcc/8.3.1
module load ffmpeg/4.2.2-gcc/8.3.1 
module load anaconda3/2019.10-gcc/8.3.1

source activate torch-1.8

cd /scratch1/zeyut/eat_detection

cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/src/extract_frames.py /scratch1/zeyut/eat_detection
cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/src/process_frames.py /scratch1/zeyut/eat_detection
cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/src/constants.py /scratch1/zeyut/eat_detection
python process_frames.py 16 1
python process_frames.py 6 1




  