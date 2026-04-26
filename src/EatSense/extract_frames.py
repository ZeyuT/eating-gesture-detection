import sys
import os
import subprocess
import numpy as np
import cv2
import glob 
import multiprocessing as mp
from constants import RAW_DATA_LOC, DATA_LOC

def ConvertMsecFormat(msec):
    return "{:02d}:{:02d}:{:.4f}".format(int(msec/(60*60*1000)),int(msec%(60*60*1000)/(60*1000)),msec%(60*1000)/1000)

def extract_raw_frames(args):
    curVideoPath,frame_save_loc, fps = args[0],args[1],args[2]
    if not os.path.exists(frame_save_loc):
        os.mkdir(frame_save_loc)
            
    # get video's duration, in ms
    query = "ffprobe -i {} -show_entries format=duration -v quiet -of csv='p=0'".format(curVideoPath)
    popen = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE)
    response = popen.stdout.read()
    popen.terminate()
    print("Processing: {}".format(curVideoPath))
    sys.stdout.flush()
    # ffmpeg outputs duration in second, then convert it to ms
    duration = float(response.decode('ascii').split("\n")[0]) *1000 
    frameCount = 1
    timestamp = 0.0
    while timestamp < duration:
        # -y: overwrite existing files without asking
        # -ss: seek to the timestamp
        query = "ffmpeg -y -ss {}ms -i {} -frames:v 1 -v quiet {}/raw_frame_{:06d}.ppm".\
                  format(int(timestamp),
                  curVideoPath,
                  frame_save_loc,
                  frameCount)
        popen = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE)
        response = popen.stdout.read()
        popen.terminate()
        frameCount += 1
        timestamp += 1000./fps
        
    return frameCount
  
if __name__ == "__main__": 

    fps = int(sys.argv[1]) 

    global RAW_FRAME_LOC
    RAW_FRAME_LOC = os.path.join(DATA_LOC, f'eatSense_rawFrames_{fps}hz/') 
    try:
        os.makedirs(RAW_FRAME_LOC,exist_ok=True)
    except:
        pass
    try:
        mp.set_start_method('fork')
    except:
        pass
    filelists = glob.glob(os.path.join(RAW_DATA_LOC,'all_RGB','*.mp4'))
    
    video_num = 0
    extract_frames_args = []
    for file_loc in filelists:
        frame_save_loc = os.path.join(RAW_FRAME_LOC,file_loc.split("/")[-1].split(".")[0])            
        extract_frames_args.append([file_loc,frame_save_loc, fps])
        video_num += 1
    
        # stop condition for debugging
        '''
        if video_num >= 5:
            break
        '''
    
    # load video file
    # zeyut: launching processes via subprocess is very unefficient and stupid,
    #        because essentially we are launching twice as many processes as you think you are
    #        But given the large computation resource from HPC, we go this way as it is.
    pool = mp.Pool(mp.cpu_count()//2-2)
    ret = pool.map(extract_raw_frames,extract_frames_args)
    pool.close()  
    pool.join()                            
    '''
    ret = []
    for arg in extract_frames_args:
        ret.append(extract_raw_frames(arg))
    '''
    