VIDEO=$1
cut -f2- /scratch1/zeyut/eat_detection/VideoData/$VIDEO/gt_frame.txt | sort | uniq -c
