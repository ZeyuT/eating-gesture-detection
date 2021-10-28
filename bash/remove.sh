#/bin/bash
#PBS -N zeyut
#PBS -l select=1:ncpus=8:mem=100gb:interconnect=any
#PBS -l walltime=12:00:00
#PBS -e /home/zeyut/eat_detection/workspace/job_output
#PBS -o /home/zeyut/eat_detection/workspace/job_output
#PBS -j oe

rm -r /scratch1/zeyut/eat_detection/trash/
