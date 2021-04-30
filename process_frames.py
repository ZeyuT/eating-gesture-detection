import sys
import os
import subprocess
import numpy as np
import cv2
import multiprocessing as mp
from constants import WIDTH,HEIGHT,RAW_FRAME_LOC,FRAME_LOC

RAW_DATA_LOC = "/home/zeyut/eat_detection/CafeteriaData/"

PRE_INTAKE_DURATION = 500 # in ms
AFTER_INTAKE_DURATION = 8000 # in ms
MAX_VIDEO = 300

    
def process_frames(args): 
    window_loc,raw_frame_loc,frame_save_loc,gt_path,video_sync_offset,fps = \
    args[0],args[1],args[2],args[3],args[4],args[5]  

    gesture_gt_loc = gt_path + '/gesture_union.txt'
    intake_gt_loc = gt_path + '/gt_union.txt' 
    gesture_starts, gesture_ends, gesture_types = load_gt(gesture_gt_loc,
                                                          intake_gt_loc,
                                                          video_sync_offset)
    
    frame_names = [f for f in os.listdir(raw_frame_loc) if f.endswith('.ppm')]
    frame_names.sort(reverse=False)

    timestamp = 0
    gesture_idx = 0
    end_timestamp = gesture_ends[0]
    f_gt_frame = open(frame_save_loc + "gt_frame_3labels.txt","w")
    #print("--------------------------------------------")
    frameNo = 1
    for frame_name in frame_names:
        # give up those frames captured before the meal started (before the first labeled gesture)
        if timestamp < gesture_starts[0]:
            timestamp += int(1 / fps * 1000)
            continue
        # give up those frames captured after the meal ended (after the last labeled gesture)
        if timestamp > gesture_ends[-1]:
            timestamp += int(1 / fps * 1000)
            continue
        
        # update gesture idx at the current time step
        while timestamp > end_timestamp + int(1 / 15 * 1000 / 2):
            # (1 / 15 * 1000) is the gt's resolution in ms. 
            # "+ int(1 / 15 * 1000 / 2)" lets the current timestamp belong to the closest gesture
            gesture_idx += 1
            end_timestamp = gesture_ends[gesture_idx]

        # write gt gesture to file
        #output_name = "frame_{:06d}.ppm".format(frameNo)
        output_name = frame_name[4:]
        f_gt_frame.write(output_name + "\t" + gesture_types[gesture_idx] + "\n")
        
        # crop and resize frames, and replace in place
        frame = cv2.imread(raw_frame_loc + frame_name, cv2.IMREAD_UNCHANGED)
        crop = frame[window_loc[1]:window_loc[3],window_loc[0]:window_loc[2],:]
        crop = cv2.resize(crop,(WIDTH,HEIGHT))
        cv2.imwrite(frame_save_loc + output_name, crop.astype(int))

        frameNo += 1
        timestamp += int(1 / fps * 1000)        
    print("{} finished: processed {} images".format(raw_frame_loc,frameNo-1))
    sys.stdout.flush()
    f_gt_frame.close()
    return frameNo
    
def load_gt(gesture_gt_loc,intake_gt_loc,video_sync_offset):
    def timeidx_to_ms(timeidx):
        return (int)(timeidx * 1000.0 / 15.0 + video_sync_offset);	# 1000/15 converts 15Hz data to milliseconds 
        
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
         
    def find_missed_gesture(gesture_locations,gesture_idx,start_time,end_time,gesture_type):
        """ find unlabeled drink/bite gestures using non-dominant hands, 
            by looking at near timestamps when using varies tools (mug, glass, etc.) from 'gt.txt' files.
            Once a unlabeled gesture is found:
            the starting timestamp is assumed to be gesture's location - PRE_INTAKE_DURATION
            the ending timestamp is assumed to be gesture's location + AFTER_INTAKE_DURATION
            Where gesture's location is the timestamp when the intaking tool touchs the subject's mouth
        """
        # if there is a unlabeled gesture with period [start_time,end_time]
        if len(gesture_locations) != 0 and start_time <= gesture_locations[gesture_idx] <= end_time:
            
            # if there is a piece in the current gesture period that is before the default starting timestamp, 
            # consider it as the original label
            if timeidx_to_ms(gesture_locations[gesture_idx]) - PRE_INTAKE_DURATION > timeidx_to_ms(start_time):                  
                gesture_types.append("non_intake")
                gesture_starts.append(timeidx_to_ms(start_time))
                gesture_ends.append(timeidx_to_ms(gesture_locations[gesture_idx]-1) - PRE_INTAKE_DURATION)
            gesture_types.append(gesture_type)          
            gesture_starts.append(max(timeidx_to_ms(gesture_locations[gesture_idx]) - PRE_INTAKE_DURATION, timeidx_to_ms(start_time))) 
            gesture_ends.append(min(timeidx_to_ms(gesture_locations[gesture_idx]) + AFTER_INTAKE_DURATION, timeidx_to_ms(end_time)))  
            # if there is a piece in the current gesture period that is after the default starting timestamp, 
            # consider it as the original label
            if timeidx_to_ms(gesture_locations[gesture_idx]) + AFTER_INTAKE_DURATION < timeidx_to_ms(end_time):                  
                gesture_types.append("non_intake")
                gesture_starts.append(timeidx_to_ms(gesture_locations[gesture_idx]+1) + AFTER_INTAKE_DURATION)
                gesture_ends.append(timeidx_to_ms(end_time))        
            return True
        else:
            return False                
                               
    bite_locations, drink_locations = load_intake_idxs(intake_gt_loc)
    
    f_gt = open(gesture_gt_loc,'r')
    is_start = 1  # a flag used to jump to the first labeled gesture in videos
    gesture_starts = []
    gesture_ends = []
    gesture_types = []
    pre_end = 0

    for line in f_gt.readlines():
        label,cur_start,cur_end = str.split(line,"\t")[0:3]
        cur_start = int(cur_start)
        cur_end = int(cur_end)
                
        #Find unlabeled video pieces
        if is_start == 0:
            if cur_start > pre_end + 1:
                unlabel_start = pre_end + 1
                unlabel_end = cur_start - 1
                # check if any drinking/eating gesture using non-dominant hand exists in rest/other gestures            
                bite_idx = 0
                drink_idx = 0
                while (1):
                    if bite_idx >= len(bite_locations)-1:
                        break
                    if bite_locations[bite_idx] < unlabel_start:
                        bite_idx += 1
                    else:
                        break
                while (1):
                    if drink_idx >= len(drink_locations)-1:
                        break
                    if drink_locations[drink_idx] < unlabel_start:
                        drink_idx += 1
                    else:
                        break
                if (not find_missed_gesture(bite_locations,bite_idx,unlabel_start,unlabel_end,"bite")) and \
                    (not find_missed_gesture(drink_locations,drink_idx,unlabel_start,unlabel_end,"drink")):
                    gesture_starts.append(timeidx_to_ms(unlabel_start)) # data index of start of gesture
                    gesture_ends.append(timeidx_to_ms(unlabel_end))	  # data index of end of gesture
                    gesture_types.append("non_intake")	# data labeled as non_intake
                    
        if label in ["rest","other"]:
            # check if any drinking/eating gesture using non-dominant hand exists in rest/other gestures            
            bite_idx = 0
            drink_idx = 0
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
            if (not find_missed_gesture(bite_locations,bite_idx,cur_start,cur_end,"bite")) and \
                    (not find_missed_gesture(drink_locations,drink_idx,cur_start,cur_end,"drink")):
                gesture_types.append("non_intake")
                gesture_starts.append(timeidx_to_ms(cur_start)) # data index of start of gesture
                gesture_ends.append(timeidx_to_ms(cur_end));  # data index of end of gesture  
        elif label in ["drink","bite"]:
            gesture_types.append(label)
            gesture_starts.append(timeidx_to_ms(cur_start)) # data index of start of gesture
            gesture_ends.append(timeidx_to_ms(cur_end));  # data index of end of gesture  
        else:
            gesture_types.append("non_intake")
            gesture_starts.append(timeidx_to_ms(cur_start)) # data index of start of gesture
            gesture_ends.append(timeidx_to_ms(cur_end));  # data index of end of gesture  
        is_start = 0
        pre_end = cur_end
 	
    f_gt.close();
    return gesture_starts, gesture_ends, gesture_types

                                   
if __name__ == "__main__": 
    fps = int(sys.argv[1])
    try:
        os.mkdir(FRAME_LOC)
    except:
        pass

    f_filelist = open(RAW_DATA_LOC + "DATA_FILENAMES.txt","r")
    f_windows = open(RAW_DATA_LOC + "window_loc.txt","r")
    windowlists = f_windows.readlines()
    f_windows.close()

    video_num = 0
    filelists = f_filelist.readlines()
    f_filelist.close()
    """
    filelists = ["p050/c1/20120228173840380.txt",
                  "p047/c2/20120229194923067.txt",
                  "p044/c1/20120301113052494.txt",
                  "p051/c1/20120312133047898.txt",
                  "p048/c1/20120223173312302.txt",
                  "p045/c2/20120309114930233.txt",
                  "p046/c1/20120222112903973.txt",
                  "p176/c1/20120328132824062.txt",
                  "p045/c1/20120309113007692.txt"]
    """
    process_frames_args = []
    for file_loc in filelists:
        syncfile_loc = RAW_DATA_LOC + "/" + str.split(file_loc,".")[0] + "_sync.txt"
        f_sync = open(syncfile_loc,"r")
        video_sync_offset = int(str.split(f_sync.readline(),"\n")[0])
        f_sync.close()

        gt_path = RAW_DATA_LOC  + "/".join(str.split(file_loc,"/")[0:-1])

        # check the integrity of raw files for the current video sample
        raw_frame_loc = RAW_FRAME_LOC + "_".join(str.split(file_loc,"/")[0:-1]) + "/"
        frame_save_loc = FRAME_LOC + "_".join(str.split(file_loc,"/")[0:-1]) + "/"
        if not os.path.exists(raw_frame_loc):
            continue
        if not os.path.exists(frame_save_loc):
            os.makedirs(frame_save_loc)
            
        window_loc = []
        for line in windowlists:
            if str.split(line,"\t")[0] == "/".join(str.split(file_loc,"/")[0:-1]):
                window_loc = list(map(int,str.split(str.split(line,"\t")[1]," ")[0:4]))
        if len(window_loc) == 0:
            continue;
        process_frames_args.append([window_loc, 
                                    raw_frame_loc, 
                                    frame_save_loc, 
                                    gt_path,
                                    video_sync_offset, 
                                    fps])

        video_num += 1
        if video_num >= MAX_VIDEO:
   			   break

    # load video file
    #pool = mp.Pool(40)
    #ret = pool.map(process_frames,process_frames_args)
    #pool.close()  
    #pool.join()                            

    video_list = [f for f in os.listdir(FRAME_LOC) if f.startswith("p")]
    video_list.sort(reverse=False)
    random_idxs = np.random.permutation(MAX_VIDEO)
    """
    Split training, validataion, testing set, with ratio being 0.7:0.15:0.15
    """
    try:
        os.mkdir(FRAME_LOC+"train_set")
    except:
        pass
    try:
        os.mkdir(FRAME_LOC+"val_set")
    except:
        pass
    try:
        os.mkdir(FRAME_LOC+"test_set")
    except:
        pass
    for idx in random_idxs[0:int(0.7*MAX_VIDEO)]:
        query = "mv {} {}".format(FRAME_LOC+video_list[idx],FRAME_LOC+"train_set")
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
    for idx in random_idxs[int(0.7*MAX_VIDEO):int(0.85*MAX_VIDEO)]:
        query = "mv {} {}".format(FRAME_LOC+video_list[idx],FRAME_LOC+"val_set")
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
    for idx in random_idxs[int(0.85*MAX_VIDEO):]:
        query = "mv {} {}".format(FRAME_LOC+video_list[idx],FRAME_LOC+"test_set")
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()        