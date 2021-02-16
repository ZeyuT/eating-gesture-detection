import sys
import os
import subprocess
import numpy as np
import cv2

raw_data_loc = "/home/zeyut/eat_detection/CafeteriaData/"
        
filelist_loc = "/home/zeyut/eat_detection/CafeteriaData/DATA_FILENAMES.txt"

save_loc = "./VideoData/"    

def ConvertMsecFormat(msec):
    return "{:02d}:{:02d}:{:02d}".format(int(msec/(60*60*1000)),int(msec%(60*60*1000)/(60*1000)),int(msec%(60*1000)/(1000)))

def extract_raw_frames(curVideoPath,frame_save_loc, fps):
    # get video's duration, in ms
    query = "ffprobe -i {} -show_entries format=duration -v quiet -of csv='p=0'".format(curVideoPath)
    response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
    print(curVideoPath)
    duration = float(response.decode('ascii').split("\n")[0]) *1000 # ffmpeg outputs duration in second, then convert it to ms
    frameNo = 1
    for timestamp in range(0,int(duration),int(1000/fps)):
        query = "ffmpeg -y -ss {} -i {} -frames:v 1 -v quiet {}frame_{:06d}.ppm".format(ConvertMsecFormat(timestamp), \
                                                                   curVideoPath,\
                                                                   frame_save_loc,\
                                                                   frameNo)
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
        frameNo += 1
        
def process_frames(window_loc, frame_save_loc, gt_loc, video_sync_offset, fps):        
    gesture_start, gesture_end, gesture_types = load_gt(gt_loc, video_sync_offset)
    
    frame_names = [f for f in os.listdir(frame_save_loc) if f.endswith('.ppm')]
    frame_names.sort(reverse=False)

    count = 0
    timestep = 0
    gesture_idx = 0
    end_timestep = gesture_end[0]

    f_gt_frame = open(frame_save_loc + "gt_frame.txt","w")
    print("--------------------------------------------")
    for frame_name in frame_names:
        # give up those frames captured before the meal started (before the first labeled gesture)
        if timestep < gesture_start[0]:
            os.remove(frame_save_loc + frame_name)
            timestep += int(1 / fps * 1000)
            continue
        # give up those frames captured after the meal started (after the last labeled gesture)
        if timestep > gesture_end[-1]:
            os.remove(frame_save_loc + frame_name)
            timestep += int(1 / fps * 1000)
            continue
        
        # update gesture idx at the current time step
        while timestep > end_timestep + int(1 / 15 * 1000 / 2):
            # (1 / 15 * 1000) is the gt's resolution in ms. 
            # "+ int(1 / 15 * 1000 / 2)" lets the current timestep belong to the closest gesture
            gesture_idx += 1
            end_timestep = gesture_end[gesture_idx]
 
        # write gt gesture to file
        f_gt_frame.write(frame_name + "\t" + gesture_types[gesture_idx] + "\n")
       
        # crop and resize frames, and replace in place
        frame = cv2.imread(frame_save_loc + frame_name, cv2.IMREAD_UNCHANGED)
        crop = frame[window_loc[1]:window_loc[3],window_loc[0]:window_loc[2],:]
        crop = cv2.resize(crop,(128,128))
        cv2.imwrite(frame_save_loc + frame_name, crop.astype(int))
        
        print("processing images: {0:05d} images finished".format(count), end="\r", flush=True)
        count += 1
        timestep += int(1 / fps * 1000)        
    print("processing images: {0:05d} images finished".format(count), flush=True)
    print("--------------------------------------------", flush=True)
    f_gt_frame.close()

def load_gt(gt_loc, video_sync_offset):
    def timeidx_to_ms(timeidx):
        return (int)(timeidx * 1000.0 / 15.0 + video_sync_offset);	# 1000/15 converts 15Hz data to milliseconds 
        
    f_gt = open(gt_loc ,'r')
    is_start = 1  # a flag used to jump to the first labeled gesture in videos
    gesture_start = []
    gesture_end = []
    gesture_types = []
    pre_end = 0
    for line in f_gt.readlines():
        label,cur_start,cur_end = str.split(line,"\t")[0:3]
        cur_start = int(cur_start)
        cur_end = int(cur_end)

        #Find unlabeled video pieces
        if is_start == 0:
            if cur_start > pre_end + 1:
                gesture_start.append(timeidx_to_ms(pre_end + 1)) # data index of start of gesture
                gesture_end.append(timeidx_to_ms(cur_start - 1))	  # data index of end of gesture
                gesture_types.append("unknown")	# data labeled as unknown
  
        gesture_types.append(label)
        gesture_start.append(timeidx_to_ms(cur_start)) # data index of start of gesture
        gesture_end.append(timeidx_to_ms(cur_end));  # data index of end of gesture  
        is_start = 0
        pre_end = cur_end
 	
    f_gt.close();
    return gesture_start, gesture_end, gesture_types
            
if __name__ == "__main__": 
    fps = int(sys.argv[1])

    try:
        os.mkdir(save_loc)
    except:
        pass

    f_filelist = open(raw_data_loc + "DATA_FILENAMES.txt","r")

    video_num = 0
    for file_loc in f_filelist.readlines():
    #file_loc = "p005/c2/20120201115556861.txt"    

        syncfile_loc = raw_data_loc + "/" + str.split(file_loc,".")[0] + "_sync.txt"
        f_sync = open(syncfile_loc,"r")
        video_sync_offset = int(str.split(f_sync.readline(),"\n")[0])
        video_name = str.split(f_sync.readline(),"\n")[0]
        f_sync.close()
        if len(str.split(video_name,".")) < 2 or not str.split(video_name,".")[1]:
            video_name = video_name + ".asf"
            
        relative_path = "/".join(str.split(file_loc,"/")[0:-1])
        video_loc = raw_data_loc + relative_path + "/" + video_name
        
        # check the integrity of files for the current video sample
        gt_loc = raw_data_loc + relative_path + '/gesture_union.txt'
        if not os.path.exists(gt_loc):
            continue
        f_windows = open(raw_data_loc + "window_loc.txt","r")
        window_loc = []
        for line in f_windows.readlines():
            if str.split(line,"\t")[0] == relative_path:
                window_loc = list(map(int,str.split(str.split(line,"\t")[1]," ")[0:4]))
        if len(window_loc) == 0:
            print("WARNING: cannot find the window loc for {}".format(relative_path))
            print("{} videos have been processed".format(video_num))
            continue;
        
        frame_save_loc = save_loc + "_".join(str.split(file_loc,"/")[0:-1]) + "/"
        if not os.path.exists(frame_save_loc):
            os.makedirs(frame_save_loc)
        
        # load video file
        extract_raw_frames(video_loc,frame_save_loc, fps)        
        process_frames(window_loc, frame_save_loc, gt_loc, video_sync_offset, fps)
           
        video_num += 1
        if video_num >= 50:
   			   break
		    
    f_filelist.close()
    f_windows.close()
    