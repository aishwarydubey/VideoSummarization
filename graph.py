import matplotlib.pyplot as plt
import numpy


def findMaxMin(arr, low, high):
    max = arr[low]
    i = low
    min=9999
    for i in range(high+1):
        if arr[i] > max:
            max = arr[i]
        if arr[i] < min:
            min = arr[i]
    return [max,min]
    
def graphTiming(timings_,scores_,videoName):
    videoName=videoName.replace(".csv","")
    videoName=videoName.replace(".wav_intensities","")
    i=videoName.rfind('/')
    print("test "+str(i))
    if(i>-1):
        video_name=videoName[(i+1):]  
    print(video_name)
    #timings_=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    #scores_=[0.0, -0.017364501953125, 0.049530029296875, 0.04852294921875, 0.0875244140625, 0.08392333984375, -0.169189453125, -0.04705810546875, -0.02001953125, -0.105621337890625,0.00482177734375, 0.115692138671875, 0.05682373046875, -0.030029296875, -0.133453369140625, -0.03851318359375, 0.068389892578125, -0.00543212890625, 0.04010009765625, -0.16070556640625, 0.05059814453125, -0.08673095703125, -0.060272216796875, 0.21697998046875, -0.105743408203125, -0.093597412109375, -0.03753662109375, -0.056549072265625,-0.01617431640625, -0.192779541015625, -0.113922119140625, 0.08612060546875, -0.040802001953125, 0.048828125, -0.0218505859375, -0.046356201171875, 0.091339111328125, -0.102508544921875, -0.11865234375, 0.0333251953125, -0.02960205078125, 0.009307861328125, -0.077850341796875, -0.01287841796875, 0.0430908203125, 0.011505126953125, -0.104583740234375, -0.16259765625, 0.068695068359375, -0.00262451171875, -0.095245361328125, 0.0479736328125, 0.06219482421875, -0.14794921875, 0.16375732421875]
    plt.plot(timings_, scores_)
    plt.grid(True)
    plt.xlabel('Timing')
    plt.ylabel('Intensity')
    plt.title('Timing Vs Sound Intensity-'+video_name)
    plt.axis([0, findMaxMin(timings_,0,len(timings_)-1)[0], findMaxMin(scores_,0,len(scores_)-1)[1]-0.10,findMaxMin(scores_,0,len(scores_)-1)[0]+0.20])
    plt.savefig('timings_vs_score_'+video_name+'_.png')
    plt.show()


    plt.xlabel('Timing')
    plt.ylabel('Intensity')
    plt.title('Timing Box Plot-'+video_name)
    plt.boxplot(scores_)
    plt.savefig('Insity_BoxPlot_'+video_name+'_.png')
    plt.show()


def graphCNN(counts,videoName):

    videoName=videoName.replace(".csv","")
    videoName=videoName.replace(".wav_intensities","")
    
    i=videoName.rfind('/')
    if(i>-1):
        video_name=videoName[(i+1):]  
    i=videoName.rfind('\\')
    if(i>-1):
        video_name=videoName[(i+1):]  
    print("video_name "+video_name)
    #counts ={'india_pak_peshdd.3gp': 8, 'india_pak_peshawards.3gp': 300, 'videoplayback_cricketq.3gp': 1, 'india_pak_.mp4': 797}
    print(counts)
    labels =counts.keys() #['Frogs', 'Hogs', 'Dogs', 'Logs']
    sizes = counts.values()#[15, 30, 45, 10]
    # only "explode" the 2nd slice (i.e. 'Hogs')
    if(len(sizes)==0):
        return
    explode = (numpy.zeros(len(sizes)))  
    explode[0]=0.2;
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.title('CNN Category Distribution-'+video_name)
    plt.savefig("CNN_Graph_%s_.png" % video_name)
    plt.tight_layout()
    
    plt.show()

