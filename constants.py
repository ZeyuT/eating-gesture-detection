FRAME_LOC = "/scratch1/zeyut/eat_detection/VideoData_224by224/"  
WIDTH = 224
HEIGHT = 224
CHANNEL = 3
LABEL_NUM = 3
LABEL_COUNTS = [421, 2318*2, 159, 342, 640]
LABEL_COUNTS_TEST = [622, 1973, 1264, 345, 584]
LABEL_TABLE = {"bite": 0, "drink": 1, "non_intake": 2}
#LABEL_TABLE =  {"bite": 0, "drink": 1, "rest": 2, "utensiling": 3, "other": 4, "unknown": 5}  
#/* 0=>up, 1=>right, 2=>down, 3=>left */ 