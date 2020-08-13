import tensorflow as tf
import numpy as np
import os,glob,cv2,time
import sys,argparse
import operator
def apply_cnn_Frames(dir):
    #dir="E:\\work\\big-project\\VideoSummarization\\dataset\\mytest"
    frames_file_path='output/'+millis+'_frames.csv'
    logfile = open('output/'+millis+'.csv','w')
    logframes = open(frames_file_path,'w')
    ## Let us restore the saved model 
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('./model/summarization-model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))
    classes = ['BaseballSwingAnalysis_swing_baseball_f_nm_np1_fr_med_17.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_f_nm_np1_fr_med_18.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_f_nm_np1_fr_med_8.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_f_nm_np1_fr_med_9.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_u_nm_np1_ba_med_0.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_u_nm_np1_ba_med_1.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_u_nm_np1_ba_med_2.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_u_nm_np1_ba_med_3.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_u_nm_np1_ba_med_4.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_u_nm_np1_ba_med_5.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_u_nm_np1_ba_med_6.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_u_nm_np1_ba_med_7.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_u_nm_np1_fr_med_10.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_u_nm_np1_fr_med_13.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_u_nm_np1_fr_med_16.avi_frames',
    'BaseballSwingAnalysis_swing_baseball_u_nm_np1_fr_med_19.avi_frames',
    'Faith_Rewarded_swing_baseball_f_cm_np1_ba_bad_11.avi_frames',
    'Faith_Rewarded_swing_baseball_f_cm_np1_ba_bad_20.avi_frames',
    'Faith_Rewarded_swing_baseball_f_cm_np1_ba_bad_35.avi_frames',
    'Faith_Rewarded_swing_baseball_f_cm_np1_le_bad_21.avi_frames',
    'Faith_Rewarded_swing_baseball_f_cm_np1_le_bad_7.avi_frames',
    'Faith_Rewarded_swing_baseball_f_cm_np1_ri_bad_27.avi_frames',
    'Faith_Rewarded_swing_baseball_f_cm_np1_ri_bad_37.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np0_fr_bad_67.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ba_bad_22.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ba_bad_44.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ba_bad_50.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ba_bad_54.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ba_bad_58.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ba_bad_60.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ba_bad_77.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ba_bad_78.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ba_med_23.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ba_med_28.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ba_med_41.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ba_med_80.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_17.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_26.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_36.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_40.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_49.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_63.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_64.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_66.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_69.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_70.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_71.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_72.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_73.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_85.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_bad_88.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_med_52.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_med_75.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_fr_med_81.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_le_bad_18.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_le_bad_25.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_le_bad_38.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_le_bad_39.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_le_bad_42.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_le_bad_47.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_le_bad_51.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_le_bad_82.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_le_med_19.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_le_med_30.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ri_bad_4.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ri_bad_53.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ri_bad_59.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ri_bad_61.avi_frames',
    'Faith_Rewarded_swing_baseball_f_nm_np1_ri_med_62.avi_frames',
    'Faith_Rewarded_swing_baseball_u_cm_np1_ri_bad_3.avi_frames',
    'Faith_Rewarded_swing_baseball_u_nm_np1_fr_bad_45.avi_frames',
    'Faith_Rewarded_swing_baseball_u_nm_np1_le_bad_12.avi_frames',
    'Faith_Rewarded_swing_baseball_u_nm_np1_ri_bad_48.avi_frames',
    'Faith_Rewarded_swing_baseball_u_nm_np1_ri_bad_57.avi_frames',
    'Hittingadouble(BXbaseball)_swing_baseball_f_nm_np1_ba_bad_0.avi_frames',
    'Hittingadouble(BXbaseball)_swing_baseball_f_nm_np1_ba_bad_1.avi_frames',
    'Hittingadouble(BXbaseball)_swing_baseball_f_nm_np1_ba_bad_2.avi_frames',
    'Hittingadouble(BXbaseball)_swing_baseball_f_nm_np1_ba_bad_3.avi_frames',
    'HittingaSingle_swing_baseball_f_cm_np1_fr_med_0.avi_frames',
    'HittingaSingle_swing_baseball_f_cm_np1_fr_med_1.avi_frames',
    'Hittingmechanics_swing_baseball_f_nm_np1_fr_bad_0.avi_frames',
    'hittingofftee2_swing_baseball_f_nm_np1_fr_med_0.avi_frames',
    'hittingofftee2_swing_baseball_f_nm_np1_fr_med_1.avi_frames',
    'hittingofftee2_swing_baseball_f_nm_np1_fr_med_10.avi_frames',
    'hittingofftee2_swing_baseball_f_nm_np1_fr_med_11.avi_frames',
    'hittingofftee2_swing_baseball_f_nm_np1_fr_med_2.avi_frames',
    'hittingofftee2_swing_baseball_f_nm_np1_fr_med_3.avi_frames',
    'hittingofftee2_swing_baseball_f_nm_np1_fr_med_4.avi_frames',
    'hittingofftee2_swing_baseball_f_nm_np1_fr_med_5.avi_frames',
    'hittingofftee2_swing_baseball_f_nm_np1_fr_med_6.avi_frames',
    'hittingofftee2_swing_baseball_f_nm_np1_fr_med_7.avi_frames',
    'hittingofftee2_swing_baseball_f_nm_np1_fr_med_8.avi_frames',
    'hittingofftee2_swing_baseball_f_nm_np1_fr_med_9.avi_frames',
    'HowtoswingaBaseballbat_swing_baseball_f_nm_np1_le_bad_0.avi_frames',
    'HowtoswingaBaseballbat_swing_baseball_f_nm_np1_le_bad_1.avi_frames',
    'longhomerunhomerunswingslowmotion_swing_baseball_f_cm_np1_ba_bad_0.avi_frames',
    'MarkTeixeiratripleagainstBoston_swing_baseball_f_cm_np1_ba_bad_0.avi_frames',
    'MattBolden(hittinginslowmotion)_swing_baseball_f_nm_np1_fr_bad_0.avi_frames',
    'MattBoldenDoubles(SlowMotion)_swing_baseball_f_cm_np1_fr_bad_0.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_11.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_12.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_13.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_14.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_15.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_16.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_17.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_18.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_19.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_2.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_3.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_4.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_5.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_6.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_7.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_8.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_9.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_u_cm_np1_fr_med_0.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_u_cm_np1_fr_med_1.avi_frames',
    'practicingmybaseballswing2009_swing_baseball_u_cm_np1_fr_med_10.avi_frames',
    'SlowMotionBPHomeRun_swing_baseball_f_nm_np1_fr_bad_0.avi_frames',
    'SlowmotionBretthitsasingle_swing_baseball_f_cm_np1_ba_bad_0.avi_frames',
    'SlowmotionBretthitsasingle_swing_baseball_f_cm_np1_ba_bad_1.avi_frames',
    'SlowMotionHomerAtOleMiss_swing_baseball_f_cm_np1_ba_bad_0.avi_frames',
    'SlowMotionofMySwing_swing_baseball_f_cm_np1_fr_bad_0.avi_frames',
    'Tannerafterwecorrected_swing_baseball_f_cm_np1_fr_bad_0.avi_frames',
    'Tannerafterwecorrected_swing_baseball_f_cm_np1_fr_bad_1.avi_frames',
    'Tannerafterwecorrected_swing_baseball_f_cm_np1_fr_bad_2.avi_frames',
    'Tannerafterwecorrected_swing_baseball_f_cm_np1_fr_bad_3.avi_frames',
    'VenomSwingIceman_swing_baseball_f_cm_np1_fr_bad_0.avi_frames']
    num_classes = len(classes)

    x = [os.path.join(r,file) for r,d,f in os.walk(dir) for file in f]
    x.sort(key=os.path.getmtime)
    # First, pass the path of the image
    millis = str(round(time.time() * 1000))


    #x.sort(key=lambda x: os.path.getmtime(x))
    for f in x:
        #dir_path = os.path.dirname(os.path.realpath(__file__))
        #print(dir_path)
        #image_path=sys.argv[1] 
        #filename = dir_path +'/' +image_path
        filename=f
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
        print(x)
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
        print("{}>{}>{}>{}>{} \n".format(filename, classes[min_index],min_value, classes[min_index1],min_value1));
        i=filename.rfind('\\')
        frameno=filename[(i+1):].rsplit(".")[0]
        s="{}>{}>{}>{}>{} \n".format(frameno, classes[min_index],min_value, classes[min_index1],min_value1);
        logfile.write(s);
        if(min_value>0.4):
            logframes.write("{}\n".format(frameno));
    logfile.close()
    logframes.close()
    return frames_file_path