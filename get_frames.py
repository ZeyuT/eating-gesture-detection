import sys
import os
import subprocess
import numpy as np
import cv2

RAW_DATA_LOC = "/home/zeyut/eat_detection/CafeteriaData/"

SAVE_LOC = "./VideoData/"    

PRE_INTAKE_DURATION = 500 # in ms
AFTER_INTAKE_DURATION = 8000 # in ms

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

def load_intake_idxs(intake_gt_loc):
    bite_locations = []
    drink_locations = []
    f_intake_gt = open(intake_gt_loc, "r")

    for line in f_intake_gt.readlines():
        cur_container = str.split(line, "\t")[4]
        if cur_container == '':
            continue
        cur_idx = int(str.split(line, "\t")[1])
        if cur_container in ["bowl","plate"]:
            bite_locations.append(cur_idx)
        elif cur_container in ["mug","glass"]:
            drink_locations.append(cur_idx)
    f_intake_gt.close()
    
    return bite_locations, drink_locations
    
def process_frames(window_loc, frame_save_loc, gesture_gt_loc, video_sync_offset, fps):        
    gesture_start, gesture_end, gesture_types = load_gt(gesture_gt_loc, video_sync_offset)
    
    frame_names = [f for f in os.listdir(frame_save_loc) if f.endswith('.ppm')]
    frame_names.sort(reverse=False)

    count = 0
    timestep = 0
    gesture_idx = 0
    end_timestep = gesture_end[0]

    f_gt_frame = open(frame_save_loc + "gt_frame_3labels.txt","w")
    print("--------------------------------------------")
    for frame_name in frame_names:
        # give up those frames captured before the meal started (before the first labeled gesture)
        if timestep < gesture_start[0]:
            #os.remove(frame_save_loc + frame_name)
            timestep += int(1 / fps * 1000)
            continue
        # give up those frames captured after the meal ended (after the last labeled gesture)
        if timestep > gesture_end[-1]:
            #os.remove(frame_save_loc + frame_name)
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

def load_gt(gesture_gt_loc, video_sync_offset):
    def timeidx_to_ms(timeidx):
        return (int)(timeidx * 1000.0 / 15.0 + video_sync_offset);	# 1000/15 converts 15Hz data to milliseconds 
    
    bite_locations, drink_locations = load_intake_idxs(intake_gt_loc)
    #bite_locations, drink_locations = [], []
    
    f_gt = open(gesture_gt_loc,'r')
    is_start = 1  # a flag used to jump to the first labeled gesture in videos
    gesture_start = []
    gesture_end = []
    gesture_types = []
    pre_end = 0
    bite_idx = 0
    drink_idx = 0
    for line in f_gt.readlines():
        label,cur_start,cur_end = str.split(line,"\t")[0:3]
        cur_start = int(cur_start)
        cur_end = int(cur_end)

        #Find unlabeled video pieces
        if is_start == 0:
            if cur_start > pre_end + 1:
                gesture_start.append(timeidx_to_ms(pre_end + 1)) # data index of start of gesture
                gesture_end.append(timeidx_to_ms(cur_start - 1))	  # data index of end of gesture
                gesture_types.append("non_intake")	# data labeled as unknown

        # check if any drinking/eating gesture using non-dominant hand exists in rest/other gestures            
        while (1):
            if bite_idx >= len(bite_locations)-1:
                break
            if bite_locations[bite_idx] < cur_start:
                bite_idx += 1
            else:
                break
        while (1):
            if drink_idx >= len(drink_locations)-1:
                break
            if drink_locations[drink_idx] < cur_start:
                drink_idx += 1
            else:
                break
        if label in ["rest","other"]:
            if len(bite_locations) != 0 and bite_locations[bite_idx] >= cur_start and bite_locations[bite_idx] <= cur_end:
                gesture_types.append("bite")
                gesture_start.append(max(timeidx_to_ms(bite_locations[bite_idx]) - PRE_INTAKE_DURATION, timeidx_to_ms(cur_start))) # data index of start of gesture
                gesture_end.append(min(timeidx_to_ms(bite_locations[bite_idx]) + AFTER_INTAKE_DURATION, timeidx_to_ms(cur_end)))  # data index of end of gesture  
            elif len(drink_locations) != 0 and drink_locations[drink_idx] >= cur_start and drink_locations[drink_idx] <= cur_end:
                gesture_types.append("drink")
                gesture_start.append(max(timeidx_to_ms(drink_locations[drink_idx]) - PRE_INTAKE_DURATION, timeidx_to_ms(cur_start))) # data index of start of gesture
                gesture_end.append(min(timeidx_to_ms(drink_locations[drink_idx]) + AFTER_INTAKE_DURATION, timeidx_to_ms(cur_end)))  # data index of end of gesture  
            else:
                gesture_types.append("non_intake")
                gesture_start.append(timeidx_to_ms(cur_start)) # data index of start of gesture
                gesture_end.append(timeidx_to_ms(cur_end));  # data index of end of gesture  
        elif label in ["drink","bite"]:
            gesture_types.append(label)
            gesture_start.append(timeidx_to_ms(cur_start)) # data index of start of gesture
            gesture_end.append(timeidx_to_ms(cur_end));  # data index of end of gesture  
        else:
            gesture_types.append("non_intake")
            gesture_start.append(timeidx_to_ms(cur_start)) # data index of start of gesture
            gesture_end.append(timeidx_to_ms(cur_end));  # data index of end of gesture  
            
        is_start = 0
        pre_end = cur_end
 	
    f_gt.close();
    return gesture_start, gesture_end, gesture_types
            
if __name__ == "__main__": 
    fps = int(sys.argv[1])

    try:
        os.mkdir(SAVE_LOC)
    except:
        pass

    f_filelist = open(RAW_DATA_LOC + "DATA_FILENAMES.txt","r")
    #f_hand_log = open("left_hand.txt","w")
    video_num = 0
    filelists = f_filelist.readlines()
    #filelists = ["p012/c2/20120203173709955.txt"]
    for file_loc in filelists:

        syncfile_loc = RAW_DATA_LOC + "/" + str.split(file_loc,".")[0] + "_sync.txt"
        f_sync = open(syncfile_loc,"r")
        video_sync_offset = int(str.split(f_sync.readline(),"\n")[0])
        video_name = str.split(f_sync.readline(),"\n")[0]
        f_sync.close()
        if len(str.split(video_name,".")) < 2 or not str.split(video_name,".")[1]:
            video_name = video_name + ".asf"
            
        relative_path = "/".join(str.split(file_loc,"/")[0:-1])
        video_loc = RAW_DATA_LOC + relative_path + "/" + video_name
        
        # check the integrity of files for the current video sample
        gesture_gt_loc = RAW_DATA_LOC + relative_path + '/gesture_union.txt'
        intake_gt_loc = RAW_DATA_LOC + relative_path + '/gt_union.txt' 

        if not os.path.exists(gesture_gt_loc):
            continue
        if not os.path.exists(intake_gt_loc):
            continue
            
        ''' 
        # check the dominant hand. Ignore people using left hands.
        intake_gt_loc = RAW_DATA_LOC + relative_path + '/gt_union.txt'
        if not os.path.exists(intake_gt_loc):
            print("no such path: {}".format(intake_gt_loc))
            continue
        f_hand = open(intake_gt_loc, "r")
        left_hand_count = 0
        total_count = 0
        for line in f_hand.readlines():
            cur_hand = str.split(line, "\t")[2]
            if cur_hand == "left":
                left_hand_count += 1
            total_count += 1
        if left_hand_count > 0.5 * total_count:
            print("left hand is dominant: {}".format(intake_gt_loc))
            f_hand_log.write(str.split(file_loc,"/")[0] + "/" + str.split(file_loc,"/")[1] + "\n")
            continue
        f_hand.close()
        '''
        f_windows = open(RAW_DATA_LOC + "window_loc.txt","r")
        window_loc = []
        for line in f_windows.readlines():
            if str.split(line,"\t")[0] == relative_path:
                window_loc = list(map(int,str.split(str.split(line,"\t")[1]," ")[0:4]))
        if len(window_loc) == 0:
            #print("WARNING: cannot find the window loc for {}".format(relative_path))
            #print("{} videos have been processed".format(video_num))
            continue;
        
        frame_save_loc = SAVE_LOC + "_".join(str.split(file_loc,"/")[0:-1]) + "/"
        if not os.path.exists(frame_save_loc):
            os.makedirs(frame_save_loc)
        
        # load video file
        extract_raw_frames(video_loc,frame_save_loc, fps)        
        process_frames(window_loc, frame_save_loc, gesture_gt_loc, video_sync_offset, fps)
           
        video_num += 1
        if video_num >= 50:
   			   break
    #f_hand_log.close()
    f_filelist.close()
    f_windows.close()
    