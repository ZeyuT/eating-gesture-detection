import sys
import os
import subprocess
import numpy as np
import cv2
from collections import defaultdict
import multiprocessing as mp
import glob
import pandas

from constants import IMAGE_SIZES,DATA_LOC,RAW_DATA_LOC
import gc

    
def process_frames(args): 
    video_frame_loc,frame_save_loc,gt_path,fps = args[0],args[1],args[2],args[3]
    
    WIDTH,HEIGHT = IMAGE_SIZES[fps]
    if not os.path.exists(frame_save_loc):
        os.makedirs(frame_save_loc)
    
    csv_df = pandas.read_csv(gt_path)
    frame_gt = csv_df['Action'].to_numpy()
    raw_fps = csv_df['fps'][0]
    raw_timestamps = csv_df['Imgs'].to_numpy().astype(float)*1000./raw_fps

    gesture_starts, gesture_ends, gesture_types = load_gt_boundaries(frame_gt, raw_timestamps)
    
    # write gesture-wise gt to file with target fps. Only write intake gestures' boundaries.
    f_gt_ges = open(os.path.join(frame_save_loc,'gt_ges_3labels.txt'),'w')
    f_gt_ges.write('\n'.join(['{}\t{}\t{}'.format(ges_type,round(start*fps/1000.),round(end*fps/1000.)) \
                  for ges_type,start,end in (zip(gesture_types,gesture_starts,gesture_ends))]))
    f_gt_ges.close()
    
    frame_names = [f for f in os.listdir(video_frame_loc) if f.endswith('.ppm')]
    frame_names.sort(reverse=False)
    timestamp = 0.0
    gesture_idx = 0
    f_gt_frame = open(os.path.join(frame_save_loc,'gt_frame_3labels.txt'),'w')
    #print('--------------------------------------------')
    frameNo = 1
    for frame_name in frame_names:
        if timestamp > raw_timestamps[-1]:
            break
        # update gesture idx to match current time step
        if timestamp > gesture_ends[gesture_idx] and gesture_idx < len(gesture_ends)-1:
            gesture_idx += 1
        
        if timestamp >= gesture_starts[gesture_idx] and timestamp <= gesture_ends[gesture_idx]:
            cur_ges = gesture_types[gesture_idx]
        else:
            cur_ges = 'non_intake'
        # write frame-wise gt to file
        output_name = frame_name[4:]
        f_gt_frame.write(output_name + '\t' + cur_ges + '\n')
        
        # crop and resize frames
        frame = cv2.imread(os.path.join(video_frame_loc,frame_name), cv2.IMREAD_UNCHANGED)
        frame = cv2.resize(frame,(WIDTH,HEIGHT))
        cv2.imwrite(os.path.join(frame_save_loc,output_name), frame.astype(int))
        gc.collect() 
        frameNo += 1
        timestamp += 1000.0 / fps        
        
        
    print('{} finished: processed {} images'.format(video_frame_loc,frameNo-1))
    sys.stdout.flush()
    f_gt_frame.close()
    
    del gesture_starts, gesture_ends
    del f_gt_ges, frame_name, frame_names, output_name, f_gt_frame
    gc.collect() 
      
    return gesture_types.count('bite'), gesture_types.count('drink')

def load_gt_boundaries(frame_gt, timestamps):
    ''' Identify temporal boundaries ('eat it' and 'drink') from frame-wise labels '''
    ''' Unit: ms '''
    gesture_starts, gesture_ends, gesture_types = [], [], []
    idx = 0
    while idx < len(frame_gt):
        if frame_gt[idx] == 'eat it':
            gesture_starts.append(timestamps[idx]) 
            gesture_types.append('bite')
            while idx < len(frame_gt) and frame_gt[idx] == 'eat it':
                idx += 1
            gesture_ends.append(timestamps[idx-1])
        elif frame_gt[idx] == 'drink':
            gesture_starts.append(timestamps[idx]) 
            gesture_types.append('drink')
            while idx < len(frame_gt) and frame_gt[idx] == 'drink':
                idx += 1
            gesture_ends.append(timestamps[idx-1])
        else:
            idx += 1
    return gesture_starts, gesture_ends, gesture_types
                        

def move_subject_videos(subject_name, target_loc):        
    for video_name in [f for f in os.listdir(FRAME_LOC) if f.startswith(subject_name)]:
        move_video(os.path.join(FRAME_LOC,video_name), target_loc)

def move_video(source_loc,target_loc):
    source_loc = source_loc.split('\n')[0]
    query = f"mv {source_loc} {target_loc}"
    popen = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE)
    response = popen.stdout.read()
    popen.terminate()
    
if __name__ == '__main__': 
    fps = int(sys.argv[1])

    global WIDTH, HEIGHT, RAW_FRAME_LOC, FRAME_LOC, SPLIT_LIST_LOC
    WIDTH,HEIGHT = IMAGE_SIZES[fps]
    RAW_FRAME_LOC = os.path.join(DATA_LOC, f'eatSense_rawFrames_{fps}hz/')  
    FRAME_LOC = os.path.join(DATA_LOC, f'VideoData_eatSense_{fps}hz/')  
    SPLIT_LIST_LOC = '/home/zeyut/meta/workspace/eating-gesture-detection/eatSense_split_record/'
    
    try:
        os.mkdir(FRAME_LOC)
    except:
        pass
    try:
        mp.set_start_method('forkserver')
    except:
        pass

    filelists = glob.glob(os.path.join(RAW_FRAME_LOC,'202*'))
        
    if 'train_set' in os.listdir(FRAME_LOC):
        dataset_exist = True
        # for modify pre-built dataset
        train_list = [f for f in os.listdir(os.path.join(FRAME_LOC,'train_set')) if f.startswith('202')]
        val_list = [f for f in os.listdir(os.path.join(FRAME_LOC,'val_set')) if f.startswith('202')]
        test_list = [f for f in os.listdir(os.path.join(FRAME_LOC,'test_set')) if f.startswith('202')]
    else:
        dataset_exist = False
        
    process_frames_args = []
    video_num = 0
    for file_loc in filelists:
        video_idx = file_loc.split("/")[-1]
        gt_path = os.path.join(RAW_DATA_LOC, 'all_2d3d_true', video_idx+'.csv')
        video_frame_loc = os.path.join(RAW_FRAME_LOC,video_idx)
        # build dataset from scratch
        frame_save_loc = os.path.join(FRAME_LOC,video_idx)
        if dataset_exist:
            # modify pre-built dataset
            if video_idx in train_list:
                frame_save_loc = os.path.join(FRAME_LOC,'train_set',video_idx)
            if video_idx in val_list:
                frame_save_loc = os.path.join(FRAME_LOC,'val_set',video_idx)
            if video_idx in test_list:
                frame_save_loc = os.path.join(FRAME_LOC,'test_set',video_idx)
            
        process_frames_args.append([video_frame_loc, 
                                    frame_save_loc, 
                                    gt_path,
                                    fps])

        video_num += 1
        
    # results = []
    # for args in process_frames_args:
    #     results.append(process_frames(args))
    #     gc.collect()
    # Parallelize the process_frames for loop
    # pool = mp.Pool(mp.cpu_count()-2)
    # results = pool.map(process_frames,process_frames_args)    
    # pool.close()  
    # pool.join()                
 
    # print(f'# bites {np.sum(results,axis=0)[0].astype(int)}  # drinks: {np.sum(results,axis=0)[1].astype(int)}')    

    '''
    Split training, validataion, testing set, with ratio being 0.7:0.15:0.15
    '''
    try:
        os.makedirs(os.path.join(FRAME_LOC,'train_set'), exist_ok=True)
    except:
        pass
    try:
        os.makedirs(os.path.join(FRAME_LOC,'val_set'), exist_ok=True)
    except:
        pass
    try:
        os.makedirs(os.path.join(FRAME_LOC,'test_set'), exist_ok=True)
    except:
        pass
    try:
        os.makedirs(SPLIT_LIST_LOC, exist_ok=True)
    except:
        pass
    if 'trainlist.txt' in os.listdir(SPLIT_LIST_LOC):
        # if there is pre-generated video list for splitting dataset
        f_trainlist = open(os.path.join(SPLIT_LIST_LOC,'trainlist.txt'),'r')
        f_vallist = open(os.path.join(SPLIT_LIST_LOC,'vallist.txt'),'r')
        f_testlist = open(os.path.join(SPLIT_LIST_LOC,'testlist.txt'),'r')    
        for video_idx in f_trainlist.readlines():
            move_video(os.path.join(FRAME_LOC,video_idx), 
                        os.path.join(FRAME_LOC,'train_set'))
        for video_idx in f_vallist.readlines():
            move_video(os.path.join(FRAME_LOC,video_idx), 
                        os.path.join(FRAME_LOC,'val_set'))
        for video_idx in f_testlist.readlines():
            move_video(os.path.join(FRAME_LOC,video_idx), 
                        os.path.join(FRAME_LOC,'test_set'))  
        f_trainlist.close()
        f_vallist.close()
        f_testlist.close()            
   
    else:
        video_list = [f for f in os.listdir(FRAME_LOC) if f.startswith('202')]
        video_list.sort(reverse=False)
        video_num = len(video_list)
        
        # randomly generate video list for splitting dataset
        f_trainlist = open(os.path.join(SPLIT_LIST_LOC,'trainlist.txt'),'w')
        f_vallist = open(os.path.join(SPLIT_LIST_LOC,'vallist.txt'),'w')
        f_testlist = open(os.path.join(SPLIT_LIST_LOC,'testlist.txt'),'w')   

        random_idxs = np.random.permutation(len(video_list))        
        for idx in random_idxs[0:int(0.7*video_num)]:
            move_video(os.path.join(FRAME_LOC,video_list[idx]), 
                        os.path.join(FRAME_LOC,'train_set'))
            f_trainlist.write(video_list[idx]+'\n')  
        for idx in random_idxs[int(0.7*video_num):int(0.85*video_num)]:
            move_video(os.path.join(FRAME_LOC,video_list[idx]), 
                        os.path.join(FRAME_LOC,'val_set'))
            f_vallist.write(video_list[idx]+'\n')  
        for idx in random_idxs[int(0.85*video_num):]:
            move_video(os.path.join(FRAME_LOC,video_list[idx]), 
                        os.path.join(FRAME_LOC,'test_set'))  
            f_testlist.write(video_list[idx]+'\n') 
        f_trainlist.close()
        f_vallist.close()
        f_testlist.close()            
