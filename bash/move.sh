#/bin/bash
#PBS -N zeyut
#PBS -l select=1:ncpus=4:mem=50gb:interconnect=any
#PBS -l walltime=24:00:00
#PBS -e /home/zeyut/eat_detection/workspace/job_output
#PBS -o /home/zeyut/eat_detection/workspace/job_output
#PBS -j oe

cp -r /scratch1/zeyut/eat_detection/VideoData_independent_8hz/ /zfs/mhealth/zeyut/eat_detection/
