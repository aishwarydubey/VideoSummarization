import os,glob,cv2,time
basepath = 'E:\\work\\big-project\\VideoSummarization\\dataset\\output'
for fname in os.listdir(basepath):
    path = os.path.join(basepath, fname)
    print(path)
    if os.path.isdir(path):
        # skip directories
        continue