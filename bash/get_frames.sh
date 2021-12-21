#/bin/bash
#PBS -N zeyut
#PBS -l select=1:ncpus=20:mem=50gb:interconnect=any
#PBS -l walltime=48:00:00
#PBS -e /home/zeyut/eat_detection/workspace/job_output
#PBS -o /home/zeyut/eat_detection/workspace/job_output
#PBS -j oe
#PBS -J 1-4

module load anaconda3/5.0.1-gcc/8.3.1
module load ffmpeg/4.2.2-gcc/8.3.1 
module load anaconda3/2019.10-gcc/8.3.1

source activate torch-1.8

cd /scratch1/zeyut/eat_detection

cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/src/extract_frames.py /scratch1/zeyut/eat_detection
cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/src/process_frames.py /scratch1/zeyut/eat_detection
cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/src/constants.py /scratch1/zeyut/eat_detection
cp /home/zeyut/eat_detection/workspace/eating-gesture-detection/bash/inputs_get_frames.txt /scratch1/zeyut/eat_detection

inputs=( $(sed -n ${PBS_ARRAY_INDEX}p inputs_get_frames.txt) )

python ./extract_frames.py ${inputs[0]}
python ./process_frames.py ${inputs[0]} ${inputs[1]}



  