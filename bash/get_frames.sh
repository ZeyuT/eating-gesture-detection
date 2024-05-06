#/bin/bash
#PBS -N zeyut
#PBS -l select=1:ncpus=32:mem=500gb:interconnect=any
#PBS -l walltime=12:00:00
#PBS -e /home/zeyut/eat_detection/workspace/job_output
#PBS -o /home/zeyut/eat_detection/workspace/job_output
#PBS -j oe

module load anaconda3/2022.05-gcc/9.5.0
module load ffmpeg/4.4.1-gcc/9.5.0

source activate torch-1.8

cd /home/zeyut/eat_detection/workspace/eating-gesture-detection/


inputs=( $(sed -n ${PBS_ARRAY_INDEX}p ./bash/inputs_get_frames.txt) )

# python ./src/extract_frames.py ${inputs[0]}
# python ./src/process_frames.py ${inputs[0]} ${inputs[1]}

python ./src/extract_frames.py 8
# cp -r /zfs/mhealth/zeyut/eat_detection/VideoData_rawFrames_8hz/ /scratch1/zeyut/eat_detection/
python ./src/process_frames.py 8 1