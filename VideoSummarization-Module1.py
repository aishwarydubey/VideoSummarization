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
import cnn_classes
sys.path.append(".\\LSTM")
import string
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread
import timeline
import graph
classes=cnn_classes.classes
FPS=20

bln=1

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
    #_time.getCurrentTime()
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
 
apply_summarization(VIDEO_INPUT_DIR)
