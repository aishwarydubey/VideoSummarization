import subprocess
import _thread
import time
import tensorflow as tf
import numpy as np
import os,glob,cv2,time
import sys,argparse
import operator
import cv2
import wave
import struct

import string
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread
import timeline
import graph
import _time
classes=cnn_classes.classes
FPS=20

bln=1
def filter_frame_nos(fname ):
    print(fname)
    _time.getCurrentTime()
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    #print(content)
    frameNos=[]
    scores=[]
    max=0.0
    for e in content:
        frameNoScore=e.split(",")
        score=float(frameNoScore[1])
        if score > max:
            max=score

    print("max "+str(max))     
    max=max-20*max/100 
    
    for e in content:
        try:
            frameNoScore=e.split(",")
            #print(frameNoScore)
            score=float(frameNoScore[1])
            if score > max:
                frameNos.append(int(frameNoScore[0]))
                scores.append(score)
        except: 
            print("Error in ")
    #print(frameNos)
    #print(scores)
	# you may also want to remove whitespace characters like `\n` at the end of each line

	#print(content)
    prev=frameNos[0];
    arr = []
    start=-1
    end=-1
    totalFrames=0
    for j in range(1,len(frameNos)-1):
        current=frameNos[j]
        next=frameNos[j+1]
        diffNext=abs(next-current)
        diffPrev=abs(current-prev)
        if diffNext<5 or diffPrev<5:
            if(start==-1):
                start=current
            else:
                end=current
        else:
            if end-start >10: 
                arr.append([start,end])
                totalFrames=totalFrames+(end-start)+1
            start=-1
            end=-1
        prev=current
    print(arr)
    print(totalFrames)
    return arr

def create_video(videoFramesDir,startEndArray,video_name):
    print(videoFramesDir)
    img1 = cv2.imread(videoFramesDir+"1.png")
    height , width , layers =  img1.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), FPS,(width,height))
    print("Creating Video")
    for startEnd in startEndArray:
        for row in range(startEnd[0], startEnd[1]):
            img1 = cv2.imread(videoFramesDir+'/'+str(row)+'.png')
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,200)
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2
            cv2.putText(img1,str(row), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
            video.write(img1)

    cv2.destroyAllWindows()
    video.release()

def apply_cnn_Frames(dir):
    print("Applying CNN Frames classification Start "+dir)
    print(dir)
    #i=dir.rfind('/')
    frameno=getName(dir)
    frames_file_path='dataset/output/'+frameno+'_frames.csv'
    logfile = open('dataset/output/'+frameno+'.csv','w')  
    # First, pass the path of the image
    millis = str(round(time.time() * 1000))
    #dir="E:\\work\\big-project\\VideoSummarization\\dataset\\mytest"
    logframes = open(frames_file_path,'w')
    ## Let us restore the saved model 
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('./model/summarization-model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))
  
    num_classes = len(cnn_classes.classes)

    x = [os.path.join(r,file) for r,d,f in os.walk(dir) for file in f]
    x.sort(key=os.path.getmtime)

    #print(x)
    #x.sort(key=lambda x: os.path.getmtime(x))
    for f in x:
        #dir_path = os.path.dirname(os.path.realpath(__file__))
        #print(f)
        #image_path=sys.argv[1] 
        #filename = dir_path +'/' +image_path
        
        filename=f
        #print(filename)
        #continue;
        image_size=128
        num_channels=3
        images = []
        # Reading the image using OpenCV
        image = cv2.imread(filename)
        # Resizing the image to our desired size and preprocessing will be done exactly as done during training
        image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0) 
        #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
        x_batch = images.reshape(1, image_size,image_size,num_channels)



        # Accessing the default graph which we have restored
        graph = tf.get_default_graph()

        # Now, let's get hold of the op that we can be processed to get the output.
        # In the original network y_pred is the tensor that is the prediction of the network
        y_pred = graph.get_tensor_by_name("y_pred:0")

        ## Let's feed the images to the input placeholders
        x= graph.get_tensor_by_name("x:0") 
        #print(x)
        y_true = graph.get_tensor_by_name("y_true:0") 
        
        y_test_images = np.zeros((1, num_classes)) 


        ### Creating the feed_dict that is required to be fed to calculate y_pred 
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result=sess.run(y_pred, feed_dict=feed_dict_testing)
        r=",".join(map(str, result));
        # result is of this format [probabiliy_of_rose probability_of_sunflower]
        #print(result[0])
        min_index, min_value = max(enumerate(result[0]), key=operator.itemgetter(1))
        result[0][min_index]=0;
        min_index1, min_value1 = max(enumerate(result[0]), key=operator.itemgetter(1))
        #print(min_index)
        #print(min_value)
        
        #print("{}>{}>{}>{}>{} \n".format(filename, classes[min_index],min_value, classes[min_index1],min_value1));
        i=filename.rfind('\\')
        frameno=filename[(i+1):].rsplit(".")[0]
        s="{}>{}>{}>{}>{} \n".format(frameno, classes[min_index],min_value, classes[min_index1],min_value1);
        logfile.write(s);
        logframes.write("{},{}\n".format(frameno,min_value));
    logfile.close()
    logframes.close()
    print("Applying CNN Frames classification En")
    return frames_file_path
def apply_ffmpeg(filename):   
    i=filename.rfind('\\')
    if(i==-1):
        i=filename.rfind('/')
    video_name=getName(filename)
    video_dir_name=filename[0:i]
    if getSize(AUDIO_DIR_PATH+video_name+'.wav')==False:
        subprocess.call('ffmpeg -i \"'+filename+'\" -vn -n -acodec pcm_s16le -ar 44100 -ac 1 \"'+AUDIO_DIR_PATH+video_name+'.wav\"', shell=True)
    print("ApplyFFMPEG "+VIDEO_FRAMES_DIR+video_name+"==="+str(getSize(VIDEO_FRAMES_DIR+video_name)))
    if getSize(VIDEO_FRAMES_DIR+video_name)==False:
        if not os.path.exists(VIDEO_FRAMES_DIR+video_name):
            os.makedirs(VIDEO_FRAMES_DIR+video_name)
        subprocess.call('ffmpeg -i \"'+filename+'\" -n -r '+str(FPS)+'/1 \"'+VIDEO_FRAMES_DIR+video_name+'\%d.png\"', shell=True)
    
    
    #"ffmpeg -i \""+videoFilePath+"\" -vn -n -acodec pcm_s16le -ar 44100 -ac 1 \""+AUDIO_OUTPUT_DIR+f.getName()+".wav"
def generate_cnn_graph(fname):
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    #print(content)
    frameNos=[]
    counts = dict()
    max=0.0
    for e in content:
        frameNoScore=e.split(">")
        if frameNoScore[1] in counts:
            counts[frameNoScore[1]] += 1
        else: 
            counts[frameNoScore[1]] = 1
    print(counts)
    graph.graphCNN(counts,fname)
def getName(video_file_cpath):
    i=video_file_cpath.rfind('\\')
    if(i==-1):
        i=video_file_cpath.rfind('/')
    video_dir_name=video_file_cpath[(i+1):]
    return video_dir_name
    
def getSize(video_file_cpath):
    if(os.path.exists(video_file_cpath)==True):
        b=0
        if(os.path.isdir(video_file_cpath)==True):
            b = len(os.listdir(video_file_cpath))
            print("DIR "+str(b))
        else:
            b = os.path.getsize(video_file_cpath)
        if(b>0):
            return True
        return False
    else:
        return False
def apply_summarization_file(video_file_cpath):
    _time.getCurrentTime()
    print("--------------------START--------"+video_file_cpath+"------------------------------------------------")
    video_dir_name=getName(video_file_cpath)
    summary_path='dataset/summary/'+video_dir_name+'_sum.avi'
    video_frames_file_path='dataset/output/'+video_dir_name
    
    global bln
    
    if getSize(summary_path) == True:
        print("\nSummary Already Exists. Skipping File! "+summary_path+"\n")
        return
    if getSize(video_frames_file_path) == False:
        print("\nVideo has NOT been extracted into frames "+video_frames_file_path+"\n")
        apply_ffmpeg(video_file_cpath)
    
    frames_file_path='dataset/output/'+video_dir_name+'_frames.csv'
    if getSize(frames_file_path) == False:
        frames_file_path=apply_cnn_Frames('dataset/output/'+video_dir_name)
    else:
        print("Skipping CNN Classification for "+video_dir_name)
    frames_file_path222='dataset/output/'+video_dir_name+'.csv'
    if getSize(frames_file_path222) == True:
        generate_cnn_graph(frames_file_path222)
    
    #frames_file_path='dataset/output/videoplayback_frames.csv'
    arrTimingVideo=filter_frame_nos(frames_file_path)
    audio_file_path=AUDIO_DIR_PATH+"\\"+video_dir_name+".wav";
    audio_log_file_path=wav_to_intesities(audio_file_path)
    _timings_array=filter_intesities(0.7,audio_log_file_path)
    #print(_timings_array)
    print(arrTimingVideo)
    _array=[]
    for startEnd in arrTimingVideo:
        for row in range(startEnd[0], startEnd[1]):
            _array.append(row)
    #executes LSTM for verifing time series. It outputs Good if series has a good classification
    if(bln==1):
        #CNN_tsc_main.execute_LSTM(_timings_array)
        bln=bln+1
    apply_overall_summarization(_timings_array,VIDEO_FRAMES_DIR+video_dir_name)
    print("------------------------END----"+video_file_cpath+"------------------------------------------------")
    
def apply_summarization(dir):    
    for fname in os.listdir(dir):
        f = os.path.join(dir, fname)
        print(f)
        if os.path.isdir(f):
            # skip directories
            continue
        #dir_path = os.path.dirname(os.path.realpath(__file__))
        #print(dir_path)
        #image_path=sys.argv[1] 
        #filename = dir_path +'/' +image_path
        apply_summarization_file(f)

def wav_to_intesities(wave_file):
    print(wave_file)
    audio_file_name=getName(wave_file)
    #i=wave_file.rfind('\\')
    #audio_file_name=wave_file[(i+1):]  
    frames_file_path='dataset/output/'+audio_file_name+'_intensities.csv'
    if os.path.exists(frames_file_path) == False:
        logfile = open(frames_file_path,'w')
        waveFile = wave.open(wave_file, 'r')
        length = waveFile.getnframes()
        #print(length)
        #print(waveFile.getsampwidth())
        #print("Frame rate "+str(waveFile.getframerate()))
        #print(max_nb_bit)
        #for i in range(0,length):
        #    waveData = waveFile.readframes(1)
        #    if(i%sample==0):
        #        data = struct.unpack("<h", waveData)
        #        print(str(i)+"  "+str(data[0]/max_nb_bit))
        nb_bits=waveFile.getsampwidth()*8
        max_nb_bit = float(2 ** (nb_bits - 1))
        sample=waveFile.getframerate()
        #read wave file from 0 to N Seconds
        # wave file reading interval=sample rate
        for i in range(0,length,sample):
            waveFile.setpos(i)
            waveData = waveFile.readframes(1)
            data = struct.unpack("<h", waveData)
            #print(str(i/sample)+","+str(data[0]/max_nb_bit))
            s="{},{}\n".format(str(i/sample), str(data[0]/max_nb_bit));
            logfile.write(s);
        logfile.close()
    return frames_file_path
def filter_intesities(threshold,audio_log_file):
    with open(audio_log_file) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    
    timings=[]
    scores=[]
    scores_=[]
    timings_=[]
    max=-1;
    for e in content:
        frameNoScore=e.split(",")
        score=float(frameNoScore[1])
        second=int(float(frameNoScore[0]))
        if score > max:
            max=score
    threshold=max-(max*25/100);
    print("threshold "+str(threshold))
    for e in content:
        frameNoScore=e.split(",")
        score=float(frameNoScore[1])
        second=int(float(frameNoScore[0]))
        scores_.append(score)
        timings_.append(second)
        if score > max:
            max=score
        if score > threshold:
            scores.append(score)
            timings.append(second)
    print("max "+str(max))
    print("----------------------------")
    print(timings_)
    print("----------------------------")
    print(scores_)
    graph.graphTiming(timings_,scores_,audio_log_file)
    
    return timings

def apply_overall_summarization(timings,videoFramesDir):    
    start_array=[]
    end_array=[]
    #print(timings)
    try:
        print("videoFramesDir "+videoFramesDir)
        video_name=getName(videoFramesDir)
        #i=videoFramesDir.rfind('/')
        #video_name=videoFramesDir[(i+1):]  
        img1 = cv2.imread(videoFramesDir+"/1.png")
        height , width , layers =  img1.shape
        video = cv2.VideoWriter("./dataset/summary/"+video_name+"_sum.avi", cv2.VideoWriter_fourcc(*"MJPG"), FPS,(width,height))
        print("Creating Video "+video_name)
        graphText=""
        i=1;
        for _time in timings:
            start=(_time-5)*FPS;
            end=(_time+3)*FPS;
            graphText+="Break "+str(i)+","+str(_time-5)+","+str(_time+3)+"#"
            i=i+1
            print(str(_time)+" Sec Start Frame "+str(start)+" End Frame "+str(end))
            for x in range(start, end):
                f = os.path.join(videoFramesDir, str(x)+".png")
                if os.path.isfile(f):
                    img1 = cv2.imread(videoFramesDir+'/'+str(x)+'.png')
                    #font                   = cv2.FONT_HERSHEY_SIMPLEX
                    #bottomLeftCornerOfText = (10,50)
                    #fontScale              = 1
                    #fontColor              = (255,255,255)
                    #lineType               = 2
                    #fontScale,
                    #cv2.putText(img1,str(_time)+"_"+str(x),bottomLeftCornerOfText,font, fontColor,lineType)
                    video.write(img1)
                #else:
                    #print(str(x)+".png Not found ", end=' ')
        cv2.destroyAllWindows()
        #print(graphText)
        timeline.CreateGanttChart2(graphText,video_name)
    except  IndexError: 
        pass


    video.release()
def apply_audio_analysis(audio_dir,videoFramesDir1):    
    _time.getCurrentTime()
    for fname in os.listdir(audio_dir):
        f = os.path.join(audio_dir, fname)
        print(f)
        if os.path.isfile(f)==False:
            # skip directories
            continue
        i=fname.find('.')
        video_name=fname[0:i]  
       
        apply_audio_analysis_file(video_name,f,videoFramesDir1)
        print("----------------------------------------------------------------------")
        
def apply_audio_analysis_file(video_name,f,videoFramesDir1):   
    file_path=wav_to_intesities(f)
    timings_array=filter_intesities(0.7,file_path)
    print("vIDEO path is "+videoFramesDir1+"/"+video_name);
    #_thread.start_new_thread(apply_overall_summarization,(timings_array,videoFramesDir1+"/"+video_name))
    apply_overall_summarization(timings_array,videoFramesDir1+video_name)
VIDEO_INPUT_DIR='E:\\work\\project\\VideoSummarization\\dataset\\dataset_input\\'     
VIDEO_FRAMES_DIR='E:\\work\\project\\VideoSummarization\\dataset\\output\\'     
AUDIO_DIR_PATH='E:\\work\\project\\VideoSummarization\\dataset\\dataset_audio\\'       
#file_path=wav_to_intesities('a.wav')
#file_path='dataset/output/a.wav_intensities.csv'
#fil=filter_intesities(0.7,file_path)
#print(fil)
#apply_overall_summarization(fil,videoFramesDir1)
#videoFramesDir1='E:\\work\\big-project\\VideoSummarization\\dataset\\output\\'     
apply_summarization(VIDEO_INPUT_DIR)
#apply_summarization_file("E:\\work\\big-project\\VideoSummarization\\dataset\\dataset_input\\Cricket_Dataset\\india 1.mp4")
#apply_ffmpeg("E:\\work\\big-project\\VideoSummarization\\dataset\\dataset_input\\Cricket_Dataset\\india 1.mp4")
#apply_audio_analysis(AUDIO_DIR_PATH,VIDEO_FRAMES_DIR)