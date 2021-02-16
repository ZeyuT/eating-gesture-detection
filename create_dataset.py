import cv2
import numpy as np
import sys
import os
import av
from collections import defaultdict

raw_data_loc = '/home/zeyut/eat_detection/CafeteriaData/'
        
filelist_loc = '/home/zeyut/eat_detection/CafeteriaData/DATA_FILENAMES.txt'

save_loc = './VideoData/'

labels  =  {0: "bite", 1: 'drink', 
            2: 'rest', 3: 'utensiling',
            4: 'other', 5: 'unlabeled'}    


def read_video_frame(target_sec, container, video_stream):                   
    time_base = float(video_stream.time_base)
    target_timestamp = int(target_sec / time_base) + video_stream.start_time
    container.seek(target_timestamp,any_frame=True,backward=True,stream=video_stream)
    for packet in container.demux(video_stream):
        if not packet.pts:
            continue
        elif packet.pts < target_timestamp:
            continue
        else:
            print(packet.decode()[0])
            #img = packet.decode()[0].to_ndarray(format='bgr24')
            #return img
    return None

def sample_images(file_loc,sample_interval):
    # load video file
    syncfile_loc = raw_data_loc + '/' + str.split(file_loc,'.')[0] + '_sync.txt'
    f_sync = open(syncfile_loc,'r')
    video_sync_offset = int(str.split(f_sync.readline(),'\n')[0])
    video_name = str.split(f_sync.readline(),'\n')[0]
    f_sync.close()
    if len(str.split(video_name,'.')) < 2 or not str.split(video_name,'.')[1]:
        video_name = video_name + '.asf'
    video_loc = raw_data_loc + '/'.join(str.split(file_loc,'/')[0:-1]) + '/' + video_name
    print(video_loc)
    #load gesture gt file
    categories = defaultdict(list)
    gt_loc = raw_data_loc + '/'.join(str.split(file_loc,'/')[0:-1]) + '/gesture_union.txt'
    if os.path.exists(gt_loc):
        f_gt = open(gt_loc ,'r')
    else:
        print(1)
        return '/'.join(str.split(file_loc,'/')[0:-1]) 
        
    pre_end_idx = -1
    for gt_line in f_gt.readlines():
        cur_label,start_idx,end_idx = str.split(gt_line)
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        categories[cur_label].append([start_idx,end_idx])
        # detect unlabeled pieces 
        if pre_end_idx > 0:
            if start_idx > pre_end_idx + 1:
                categories['unlabeled'].append([pre_end_idx+1,start_idx-1])
        pre_end_idx = end_idx
        
    # Locate frames and save them
    container = av.open(video_loc)
    video_stream = next(s for s in container.streams if s.type == 'video')  
    for label in labels.values():
        timeidxs = categories[label]
        cur_save_loc = save_loc + label + '/' + str.split(file_loc,'/')[-3] + '/' 
        for label_name in labels.values():
            try:
                os.mkdir(cur_save_loc)
            except:
                pass
        occur_idx = 0

        for timeidx in timeidxs:
            start_sec = timeidx[0]/15 + video_sync_offset/1000
            end_sec = timeidx[1]/15 + video_sync_offset/1000
  
            sample_idx = 0
            for cur_sec in np.arange(start_sec, end_sec+0.001, sample_interval):
                cur_frame = read_video_frame(cur_sec,container,video_stream);
                if cur_frame is not None:
                    cur_frame_name = '{}_{}_{}_{}_{}_{}.ppm'.format(str.split(file_loc,'/')[-2], occur_idx, sample_idx, timeidx[0], timeidx[1], int(1000*cur_sec))
                    cv2.imwrite(cur_save_loc + cur_frame_name,cur_frame)	
                    sample_idx += 1
            occur_idx += 1
    return 0
                                     
if __name__ == '__main__': 
    fps = float(sys.argv[1]) 
    
    sample_interval = float(1/fps)
           
    try:
        os.mkdir(save_loc)
    except:
        pass
    for label_name in labels.values():
        try:
            os.mkdir(save_loc+label_name)
        except:
            pass
        

    f_filelist = open(filelist_loc,'r')
    #for file_loc in f_filelist.readlines():
    file_loc = 'p005/c2/20120201115556861.txt'
    sample_images(file_loc,sample_interval)
    """
        # load video file
        syncfile_loc = raw_data_loc + '/' + str.split(file_loc,'.')[0] + '_sync.txt'
        f_sync = open(syncfile_loc,'r')
        video_sync_offset = int(str.split(f_sync.readline(),'\n')[0])
        video_name = str.split(f_sync.readline(),'\n')[0]
        f_sync.close()
        if len(str.split(video_name,'.')) < 2 or not str.split(video_name,'.')[1]:
            video_name = video_name + '.asf'
        video_loc = raw_data_loc + '/'.join(str.split(file_loc,'/')[0:-1]) + '/' + video_name
        
        #load gesture gt file
        categories = defaultdict(list)
        gt_loc = raw_data_loc + '/'.join(str.split(file_loc,'/')[0:-1]) + '/gesture_union.txt'
        f_gt = open(raw_data_loc + '/' + gt_loc,'r')
        pre_end_idx = -1
        for gt_line in f_gt.readlines():
            cur_label,start_idx,end_idx = str.split(gt_line)
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            categories[cur_label].append([start_idx,end_idx])
            # detect unlabeled pieces 
            if pre_end_idx > 0:
                if start_idx > pre_end_idx + 1:
                    categories['unlabeled'].append([pre_end_idx+1,start_idx-1])
            pre_end_idx = end_idx

        
        print('1')
        # Locate frames and save them
        container = av.open(video_loc)
        stream = next(s for s in container.streams if s.type == 'video')  
        for label in labels.values():
            timestamps = categories[label]
            cur_save_loc = save_loc + label + '/' + str.split(file_loc,'/')[-3] + '/' 
            for label_name in labels.values():
                try:
                    os.mkdir(cur_save_loc)
                except:
                    pass
            occur_idx = 0
            
            for timestamp in timestamps:
                start_sec = timestamp[0]/15 + video_sync_offset/1000
                end_sec = timestamp[1]/15 + video_sync_offset/1000
                
                time_idx = 0
                for cur_sec in np.arange(start_sec, end_sec+0.001, sample_interval):
                    cur_frame = read_video_frame(cur_sec,container,stream);
                    cur_frame_name = '{}_{}_{}_{}_{}_{}.jpg'.format(str.split(file_loc,'/')[-2], occur_idx, time_idx, timestamp[0], timestamp[1], int(1000*cur_sec))
                    cv2.imwrite(cur_save_loc + cur_frame_name,cur_frame)	
                    time_idx += 1
                    print(cur_save_loc + '/' + cur_frame_name)
                occur_idx += 1
                
        print('2')
    """ 
    f_filelist.close()
