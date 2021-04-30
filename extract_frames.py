import sys
import os
import subprocess
import numpy as np
import cv2
import multiprocessing as mp
from constants import WIDTH,HEIGHT,RAW_FRAME_LOC

RAW_DATA_LOC = "/home/zeyut/eat_detection/CafeteriaData/"

PRE_INTAKE_DURATION = 500 # in ms
AFTER_INTAKE_DURATION = 8000 # in ms

def ConvertMsecFormat(msec):
    return "{:02d}:{:02d}:{:02d}".format(int(msec/(60*60*1000)),int(msec%(60*60*1000)/(60*1000)),int(msec%(60*1000)/(1000)))

def extract_raw_frames(args):
    curVideoPath,frame_save_loc, fps = args[0],args[1],args[2]
    if not os.path.exists(frame_save_loc):
        os.makedirs(frame_save_loc)
            
    # get video's duration, in ms
    query = "ffprobe -i {} -show_entries format=duration -v quiet -of csv='p=0'".format(curVideoPath)
    response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
    print("Processing: {}".format(curVideoPath))
    sys.stdout.flush()
    duration = float(response.decode('ascii').split("\n")[0]) *1000 # ffmpeg outputs duration in second, then convert it to ms
    frameCount = 1
    for timestamp in range(0,int(duration),int(1000/fps)):
        query = "ffmpeg -y -ss {} -i {} -frames:v 1 -v quiet {}raw_frame_{:06d}.ppm".format(ConvertMsecFormat(timestamp), \
                                                                   curVideoPath,\
                                                                   frame_save_loc,\
                                                                   frameCount)
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
        frameCount += 1
    return frameCount
  
if __name__ == "__main__": 
    fps = int(sys.argv[1]) 
    try:
        os.mkdir(RAW_FRAME_LOC)
    except:
        pass

    f_filelist = open(RAW_DATA_LOC + "DATA_FILENAMES.txt","r")
    f_windows = open(RAW_DATA_LOC + "window_loc.txt","r")
    windowlists = f_windows.readlines()
    f_windows.close()

    video_num = 0
    filelists = f_filelist.readlines()
    f_filelist.close()

    extract_frames_args = []
    process_frames_args = []
    for file_loc in filelists:
        syncfile_loc = RAW_DATA_LOC + "/" + str.split(file_loc,".")[0] + "_sync.txt"
        f_sync = open(syncfile_loc,"r")
        video_sync_offset = int(str.split(f_sync.readline(),"\n")[0])
        video_name = str.split(f_sync.readline(),"\n")[0]
        f_sync.close()
        if len(str.split(video_name,".")) < 2 or not str.split(video_name,".")[1]:
            video_name = video_name + ".asf"
            
        relative_gt_path = "/".join(str.split(file_loc,"/")[0:-1])
        video_loc = RAW_DATA_LOC + relative_gt_path + "/" + video_name
        
        # check the integrity of files for the current video sample
        gesture_gt_loc = RAW_DATA_LOC + relative_gt_path + '/gesture_union.txt'
        intake_gt_loc = RAW_DATA_LOC + relative_gt_path + '/gt_union.txt' 

        if not os.path.exists(gesture_gt_loc):
            continue
        if not os.path.exists(intake_gt_loc):
            continue

        window_loc = []
        for line in windowlists:
            if str.split(line,"\t")[0] == relative_gt_path:
                window_loc = list(map(int,str.split(str.split(line,"\t")[1]," ")[0:4]))

        if len(window_loc) == 0:
            #print("WARNING: cannot find the window loc for {}".format(relative_gt_path))
            #print("{} videos have been processed".format(video_num))
            continue;
        
        frame_save_loc = RAW_FRAME_LOC + "_".join(str.split(file_loc,"/")[0:-1]) + "/"

            
        extract_frames_args.append([video_loc,frame_save_loc, fps])
        process_frames_args.append([window_loc, frame_save_loc, gesture_gt_loc, video_sync_offset, fps])

        video_num += 1
        if video_num >= 300:
   			   break

    # load video file
    pool = mp.Pool(40)
    ret = pool.map(extract_raw_frames,extract_frames_args)
    pool.close()  
    pool.join()                            

    