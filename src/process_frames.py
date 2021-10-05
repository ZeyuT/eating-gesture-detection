import sys
import os
import subprocess
import numpy as np
import cv2
from collections import defaultdict
import multiprocessing as mp
from constants import WIDTH,HEIGHT,RAW_FRAME_LOC,FRAME_LOC

RAW_DATA_LOC = "/home/zeyut/eat_detection/CafeteriaData/"

# version 1
PRE_INTAKE_DURATION = {"bite":1000,"drink":2067} # in ms
AFTER_INTAKE_DURATION = {"bite":1067,"drink":3200} # in ms
MAX_VIDEO = 1000

    
def process_frames(args): 
    window_loc,video_frame_loc,frame_save_loc,gt_path,video_sync_offset,fps = \
    args[0],args[1],args[2],args[3],args[4],args[5]  

    if not os.path.exists(frame_save_loc):
        os.makedirs(frame_save_loc)
            
    gesture_gt_loc = os.path.join(gt_path,"gesture_union.txt")
    intake_gt_loc = os.path.join(gt_path,"gt_union.txt")
    gesture_starts, gesture_ends, gesture_types, added_ges_durations = load_gt(gesture_gt_loc,
                                                                              intake_gt_loc,
                                                                              video_sync_offset)
    """
    # write gesture-wise gt to file. Only write intake gestures' boundaries  
    f_gt_ges = open(os.path.join(frame_save_loc,"gt_ges_3labels.txt"),"w")
    f_gt_ges.write("\n".join(["{}\t{}\t{}".format(ges_type,round(start*fps/1000),round(end*fps/1000)) \
                  for ges_type,start,end in (zip(gesture_types,gesture_starts,gesture_ends)) if ges_type != "non_intake"]))
    f_gt_ges.close()
    
    frame_names = [f for f in os.listdir(video_frame_loc) if f.endswith('.ppm')]
    frame_names.sort(reverse=False)
    timestamp = 0
    gesture_idx = 0
    end_timestamp = gesture_ends[0]
    f_gt_frame = open(os.path.join(frame_save_loc,"gt_frame_3labels.txt"),"w")
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
        
        # update gesture idx to the current time step
        while timestamp > end_timestamp + int(1 / 15 * 1000 / 2):
            # (1 / 15 * 1000) is the gt's resolution in ms. 
            # "+ int(1 / 15 * 1000 / 2)" lets the current timestamp belong to the closest gesture
            gesture_idx += 1
            end_timestamp = gesture_ends[gesture_idx]

        # write frame-wise gt to file
        #output_name = "frame_{:06d}.ppm".format(frameNo)
        output_name = frame_name[4:]
        f_gt_frame.write(output_name + "\t" + gesture_types[gesture_idx] + "\n")

        # crop and resize frames, and replace in place
        frame = cv2.imread(os.path.join(video_frame_loc,frame_name), cv2.IMREAD_UNCHANGED)
        crop = frame[window_loc[1]:window_loc[3],window_loc[0]:window_loc[2],:]
        crop = cv2.resize(crop,(WIDTH,HEIGHT))
        cv2.imwrite(os.path.join(frame_save_loc,output_name), crop.astype(int))

        frameNo += 1
        timestamp += int(1 / fps * 1000)        
  
    print("{} finished: processed {} images".format(video_frame_loc,frameNo-1))
    sys.stdout.flush()
    f_gt_frame.close()
    """  
    return list(added_ges_durations.values())
    
def load_gt(gesture_gt_loc,intake_gt_loc,video_sync_offset):
    
    """ Load ground truths from txt files. txt files only include gestures using dominant hands.
        Then find unlabeled drink/bite gestures using non-dominant hands, 
        by looking at near timestamps when using varies tools (mug, glass, etc.) from 'gt.txt' files.
        Once a unlabeled gesture is found:
        the starting timestamp is assumed to be gesture's location - PRE_INTAKE_DURATION
        the ending timestamp is assumed to be gesture's location + AFTER_INTAKE_DURATION
        Where gesture's location is the timestamp when the intaking tool touchs the subject's mouth
    """
    added_ges_durations = {"bite":0.0, "drink":0.0}
    def timeidx_to_ms(timeidx):
        return (int)(timeidx * 1000.0 / 15.0 + video_sync_offset);	# 1000/15 converts 15Hz data to milliseconds 
        
    def load_intake_idxs(intake_gt_loc):
        """
        load the central time idxs of intake gestures from gt_union.txt. time idxs are in 15 hz.
        """
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
         
    def record_missed_gesture(gesture_info,start_time,end_time):
        # create gestures for those unlabeled intake gestures, with statistically obtained pre and after offset duration.
        # consider the background as non-intake.
        cur_start_time = start_time
        for idx, intake in enumerate(gesture_info):
            pivot_time = intake[0]
            label = intake[1]
            # check if there is a piece of background before the current intake gesture 
            if pivot_time - PRE_INTAKE_DURATION[label] > cur_start_time:        
                gesture_types.append("non_intake")
                gesture_starts.append(cur_start_time)
                # move backward by 1 timestamp in 15 hz, 
                # so that the gesture_end do not overlap with the start time of the next gesture
                # the same strategy applies in the follows.
                gesture_ends.append(int(pivot_time - PRE_INTAKE_DURATION[label] - 1000/15)) 
                
            gesture_types.append(label)
            gesture_starts.append(max(pivot_time - PRE_INTAKE_DURATION[label], cur_start_time)) 
            
            if idx == len(gesture_info) - 1:
                if pivot_time + AFTER_INTAKE_DURATION[label] > end_time:
                    gesture_ends.append(end_time)
                    added_ges_durations[gesture_types[-1]] += gesture_ends[-1] - gesture_starts[-1]
                else:
                    gesture_ends.append(int(pivot_time + AFTER_INTAKE_DURATION[label] - 1000/15))
                    added_ges_durations[gesture_types[-1]] += gesture_ends[-1] - gesture_starts[-1]
                    # there is a piece of background after the last intake gesture 
                    gesture_types.append("non_intake")
                    gesture_starts.append(pivot_time + AFTER_INTAKE_DURATION[label])
                    gesture_ends.append(end_time)                   
            else:
                # meet the requirement for gestures' pre_intake_duration first.
                gesture_ends.append(min(pivot_time + AFTER_INTAKE_DURATION[label], \
                                        int(gesture_info[idx+1][0] - PRE_INTAKE_DURATION[gesture_info[idx+1][1]]- 1/15)))   
                added_ges_durations[gesture_types[-1]] += gesture_ends[-1] - gesture_starts[-1]
            cur_start_time = gesture_ends[-1]   
                 
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
        cur_end = int(str.split(cur_end,"\n")[0])

        #Find unlabeled video pieces
        if is_start == 0:
            if cur_start > pre_end + 1:
                unlabel_start = pre_end + 1
                unlabel_end = cur_start - 1
                # check if any drinking/eating gesture using non-dominant hand exists in rest/other gestures   
                # firstly, find the nearest bite&drink event after the unlabel_start          
                bite_idx = 0
                drink_idx = 0
                unlabeled_intakes = []
                while (1):
                    if bite_idx >= len(bite_locations)-1:
                        break
                    if bite_locations[bite_idx] < unlabel_start:
                        bite_idx += 1
                    elif bite_locations[bite_idx] <= unlabel_end:
                        unlabeled_intakes.append([timeidx_to_ms(bite_locations[bite_idx]),"bite"])
                        bite_idx += 1
                    else:
                        break
                while (1):
                    if drink_idx >= len(drink_locations)-1:
                        break
                    if drink_locations[drink_idx] < unlabel_start:
                        drink_idx += 1
                    elif drink_locations[drink_idx] <= unlabel_end:
                        unlabeled_intakes.append([timeidx_to_ms(drink_locations[drink_idx]),"drink"])
                        drink_idx += 1
                    else:
                        break
                
                if len(unlabeled_intakes) != 0:
                    unlabeled_intakes.sort(key=lambda x:x[0])
                    record_missed_gesture(unlabeled_intakes,timeidx_to_ms(unlabel_start),timeidx_to_ms(unlabel_end))
                else:
                    gesture_starts.append(timeidx_to_ms(unlabel_start)) # data index of start of gesture
                    gesture_ends.append(timeidx_to_ms(unlabel_end)) # data index of end of gesture
                    gesture_types.append("non_intake") # data labeled as non_intake
        if label in ["rest","other","utensiling"]:
            # check if any drinking/eating gesture using non-dominant hand exists in rest/other/utensiling gestures   
            # firstly, find the nearest bite&drink event after the unlabel_start          
            bite_idx = 0
            drink_idx = 0
            unlabeled_intakes = []
            while (1):
                if bite_idx >= len(bite_locations):
                    break
                if bite_locations[bite_idx] < cur_start:
                    bite_idx += 1
                elif bite_locations[bite_idx] <= cur_end:
                    unlabeled_intakes.append([timeidx_to_ms(bite_locations[bite_idx]),"bite"])
                    bite_idx += 1
                else:
                    break
            while (1):
                if drink_idx >= len(drink_locations):
                    break
                if drink_locations[drink_idx] < cur_start:
                    drink_idx += 1
                elif drink_locations[drink_idx] <= cur_end:
                    unlabeled_intakes.append([timeidx_to_ms(drink_locations[drink_idx]),"drink"])
                    drink_idx += 1
                else:
                    break
                
            if len(unlabeled_intakes) != 0:
                unlabeled_intakes.sort(key=lambda x:x[0])
                record_missed_gesture(unlabeled_intakes,timeidx_to_ms(cur_start),timeidx_to_ms(cur_end))   
            else:
                gesture_starts.append(timeidx_to_ms(cur_start)) # data index of start of gesture
                gesture_ends.append(timeidx_to_ms(cur_end)) # data index of end of gesture
                gesture_types.append("non_intake") # data labeled as non_intake
        else:
            # label in ["drink","bite"]
            gesture_types.append(label)
            gesture_starts.append(timeidx_to_ms(cur_start)) # data index of start of gesture
            gesture_ends.append(timeidx_to_ms(cur_end));  # data index of end of gesture  
        is_start = 0
        pre_end = cur_end
         
    f_gt.close()
    return gesture_starts, gesture_ends, gesture_types, added_ges_durations

def move_subject_videos(subject_name, target_loc):        
    for video_name in [f for f in os.listdir(FRAME_LOC) if f.startswith(subject_name)]:
        move_video(os.path.join(FRAME_LOC,video_name), target_loc)

def move_video(source_loc,target_loc):
    query = "mv {} {}".format(source_loc,target_loc)
    response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
                                           
if __name__ == "__main__": 
    fps = int(sys.argv[1])
    subject_independent = int(sys.argv[2])
    try:
        os.mkdir(FRAME_LOC)
    except:
        pass

    f_filelist = open(os.path.join(RAW_DATA_LOC,"DATA_FILENAMES.txt"),"r")
    f_windows = open(os.path.join(RAW_DATA_LOC,"window_loc.txt"),"r")
    windowlists = f_windows.readlines()
    f_windows.close()

    video_num = 0
    filelists = f_filelist.readlines()
    f_filelist.close()
    
    # for modify pre-built dataset
    train_list = [f for f in os.listdir(FRAME_LOC+"train_set") if f.startswith("p")]
    val_list = [f for f in os.listdir(FRAME_LOC+"val_set") if f.startswith("p")]
    test_list = [f for f in os.listdir(FRAME_LOC+"test_set") if f.startswith("p")]
    
    #filelists = ["p352/c3/20120503175729964.txt"]

    process_frames_args = []
    for file_loc in filelists:
        syncfile_loc = os.path.join(RAW_DATA_LOC,str.split(file_loc,".")[0] + "_sync.txt")
        f_sync = open(syncfile_loc,"r")
        video_sync_offset = int(str.split(f_sync.readline(),"\n")[0])
        f_sync.close()

        gt_path = os.path.join(RAW_DATA_LOC, "/".join(str.split(file_loc,"/")[0:-1]))
        video_idx = "_".join(str.split(file_loc,"/")[0:-1])
        video_frame_loc = os.path.join(RAW_FRAME_LOC,video_idx)
        
        # for modify pre-built dataset
        if video_idx in train_list:
            frame_save_loc = os.path.join(FRAME_LOC,"train_set",video_idx)
        if video_idx in val_list:
            frame_save_loc = os.path.join(FRAME_LOC,"val_set",video_idx)
        if video_idx in test_list:
            frame_save_loc = os.path.join(FRAME_LOC,"test_set",video_idx)
        
        #frame_save_loc = os.path.join(FRAME_LOC,video_idx)
        # check the integrity of raw files for the current video sample
        '''
        if not os.path.exists(video_frame_loc):
            continue
        '''
        window_loc = []
        for line in windowlists:
            if str.split(line,"\t")[0] == "/".join(str.split(file_loc,"/")[0:-1]):
                window_loc = list(map(int,str.split(str.split(line,"\t")[1]," ")[0:4]))
        if len(window_loc) == 0:
            continue;
            
        process_frames_args.append([window_loc, 
                                    video_frame_loc, 
                                    frame_save_loc, 
                                    gt_path,
                                    video_sync_offset, 
                                    fps])

        video_num += 1
        # stop condition for debugging
        '''
        if video_num >= MAX_VIDEO:
   			   break
        '''

    pool = mp.Pool(40)
    results = pool.map(process_frames,process_frames_args)    
    print(f"new bites& drink frames: {(np.sum(results,axis=0)*8/1000).astype(int)}")    
    pool.close()  
    pool.join()                

    """            
    video_list = [f for f in os.listdir(FRAME_LOC) if f.startswith("p")]
    video_list.sort(reverse=False)
    video_num = len(video_list)
    subject_list = np.array([str.split(video,"_")[0] for video in video_list])
    subject_set = sorted(np.unique(subject_list),reverse=False)
    subject_num = len(subject_set)
    
    '''
    Split training, validataion, testing set, with ratio being 0.7:0.15:0.15
    '''
    try:
        os.mkdir(os.path.join(FRAME_LOC,"train_set"))
    except:
        pass
    try:
        os.mkdir(os.path.join(FRAME_LOC,"val_set"))
    except:
        pass
    try:
        os.mkdir(os.path.join(FRAME_LOC,"test_set"))
    except:
        pass

    f_trainlist = open(os.path.join(FRAME_LOC,"trainlist.txt"),"w")
    f_vallist = open(os.path.join(FRAME_LOC,"vallist.txt"),"w")
    f_testlist = open(os.path.join(FRAME_LOC,"testlist.txt"),"w")   
    
    if subject_independent:
        random_idxs = np.random.permutation(subject_num)
        for idx in random_idxs[0:int(0.7*subject_num)]:
            move_subject_videos(subject_set[idx], 
                                os.path.join(FRAME_LOC,"train_set"))
            f_trainlist.write(subject_set[idx]+"\n")     
        for idx in random_idxs[int(0.7*subject_num):int(0.85*subject_num)]:
            move_subject_videos(subject_set[idx], 
                                os.path.join(FRAME_LOC,"val_set"))      
            f_vallist.write(subject_set[idx]+"\n")      
        for idx in random_idxs[int(0.85*subject_num):]:
            move_subject_videos(subject_set[idx], 
                                os.path.join(FRAME_LOC,"test_set"))        
            f_testlist.write(subject_set[idx]+"\n")        
    else:  
        random_idxs = np.random.permutation(video_list)        
        for idx in random_idxs[0:int(0.7*video_num)]:
            move_video(os.path.join(FRAME_LOC,video_list[idx]), 
                        os.path.join(FRAME_LOC,"train_set"))
            f_trainlist.write(video_list[idx]+"\n")  
        for idx in random_idxs[int(0.7*video_num):int(0.85*video_num)]:
            move_video(os.path.join(FRAME_LOC,video_list[idx]), 
                        os.path.join(FRAME_LOC,"val_set"))
            f_vallist.write(video_list[idx]+"\n")  
        for idx in random_idxs[int(0.85*video_num):]:
            move_video((os.path.join(FRAME_LOC,video_list[idx]), 
                        os.path.join(FRAME_LOC,"test_set")))  
            f_testlist.write(video_list[idx]+"\n") 
    f_trainlist.close()
    f_vallist.close()
    f_testlist.close()            
    """