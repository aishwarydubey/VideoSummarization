import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.dates
from matplotlib.dates import HOURLY,MINUTELY, DateFormatter, rrulewrapper, RRuleLocator 
import numpy as np
import datetime

now = datetime.datetime.now()  
year=now.year
month=now.month
day=now.day
def _create_date(datetxt):
    """Creates the date"""
    hour=int(datetxt)/3600
    min=int((int(datetxt)%3600)/60)
    sec=int(datetxt)%60
    date = dt.datetime(int(year), int(month), int(day),int(hour),int(min),int(sec))
    mdate = matplotlib.dates.date2num(date) 
    return mdate
def _create_dates_str(datetxt):
    """Creates the date"""
    hour=int(int(datetxt)/3600)
    min=int((int(datetxt)%3600)/60)
    sec=int(int(datetxt)%60)
    
    #hour=int(int(datetxt)/60)
 
    #min=int(int(datetxt)%60)
    s="";
    if(hour>0):
        s=str(hour)+" hour"
    if(min>0):       
        s=s+" "+str(min)+" Min"
    if(sec>0):       
        s=s+" "+str(sec)+" Sec"  
    return s
def CreateGanttChart(fname):
    """
        Create gantt charts with matplotlib
        Give file name.
    """ 

    try:
        textlist=open(fname).readlines()
        
    except:
        return
    CreateGanttChart2(textlist)
def CreateGanttChart2(a,video_name):
    video_name=video_name.replace(".csv","")
    video_name=video_name.replace(".wav_intensities","")
    ylabels = []
    customDates = []
    breaksData=[]
    textlist=a.split("#")
    textlist = [x.strip() for x in textlist] 
    for tx in textlist:
        if not tx.startswith('#'):
            if(len(tx)>0):
                print(tx+" Line")
                ylabel,startdate,enddate=tx.split(',')
                ylabels.append(ylabel.replace('\n',''))
                customDates.append([_create_date(startdate.replace('\n','')),_create_date(enddate.replace('\n',''))])
                breaksData.append([int(startdate.replace('\n','')),int(enddate.replace('\n',''))])
    ilen=len(ylabels)
    pos = np.arange(0.5,ilen*0.5+0.5,0.5)
    task_dates = {}
    for i,task in enumerate(ylabels):
        task_dates[task] = customDates[i]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    for i in range(len(ylabels)):
         start_date,end_date = task_dates[ylabels[i]]
         ax.barh((i*0.5)+0.5, end_date - start_date, left=start_date, height=0.4, align='center', edgecolor='lightgreen', color='orange', alpha = 0.8)
    for i in range(len(breaksData)):
         v=breaksData[i][1]-breaksData[i][0]
         
         print(v)
         ax.text(_create_date(breaksData[i][1] + 3), (i*0.5)+0.5, _create_dates_str(breaksData[i][0])+"-"+_create_dates_str(v), color='blue')
    
    locsy, labelsy = plt.yticks(pos,ylabels)
    plt.setp(labelsy, fontsize = 10)
#    ax.axis('tight')
    ax.set_ylim(ymin = -0.1, ymax = ilen*0.5+0.5)
    ax.grid(color = 'g', linestyle = ':')
    ax.xaxis_date()
    rule = rrulewrapper(MINUTELY, interval=10)
    loc = RRuleLocator(rule)
    #formatter = DateFormatter("%d-%b '%y")
    formatter = DateFormatter("%H:%M")
  
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(formatter)
    labelsx = ax.get_xticklabels()
    
    plt.setp(labelsx, rotation=30, fontsize=8)
 
    font = font_manager.FontProperties(size='small')
    ax.legend(loc=1,prop=font)
 
    ax.invert_yaxis()
    fig.autofmt_xdate()
    plt.title('Video Highlight Timeline - '+video_name)
    plt.xlabel('Timing')
    plt.ylabel('Break Duration')
    plt.savefig('breaks_timings.svg')
    plt.savefig('breaks_timings'+video_name+'_.png')
    plt.grid(True)
    plt.show()
 
if __name__ == '__main__':
    fname=r"timelne_input.txt"
    #CreateGanttChart(fname)
    
    
    