import numpy as np
import cv2
import time
from Utilities import flattenImages
from PIL import Image
import cntk
import os
import _cntk_py
from cntk.ops.functions import load_model
from cntk.device import set_default_device, gpu
set_default_device(gpu(0))
Paths = ["E:/img/Animals/Rabbit/","E:/img/Animals/Racoon/","E:/img/Animals/Skunk/","E:/img/Animals/Weasel/","E:/img/Animals/Bobcat/","E:/img/Animals/Cat/","E:/img/Animals/Chicken/","E:/img/Animals/Coyote/","E:/img/Animals/Deer/","E:/img/Animals/Dog/","E:/img/Animals/Duck/","E:/img/Animals/Eagle/","E:/img/Animals/Hawk/","E:/img/Animals/MountainLion/","E:/img/Animals/Owl/","E:/img/Animals/Possum/","E:/img/Animals/Robin/","E:/img/Animals/Squirrel/","E:/img/Animals/Woodpecker/","E:/img/Animals/Humans/","E:/img/Animals/BlueJay/" ,"E:/img/Animals/Backgrounds/" ]
Videos = ["ch2_223_0-140.avi","ch2_223_140_320.avi","ch2_223_320_500.avi","ch2_223_5-7.avi","ch2_223_7-9.avi","ch2_223_9-11.avi","ch2_223_11-13.avi","ch2_223_13-15.avi","ch2_223_15-17.avi","ch2_223_17-19.avi","ch2_223_19-21.avi","ch2_223_21-23.avi","ch2_223_23-01.avi","ch2_224_01-03.avi","ch2_224_03-05.avi","ch2_224_05-07.avi","ch2_224_07-09.avi","ch2_224_09-11.avi","ch2_224_11-13.avi","ch2_224_13-15.avi","ch2_224_15-17.avi","ch2_224_17-19.avi","ch2_224_19-21.avi","ch2_224_21-23.5.avi"]
cap = cv2.VideoCapture(0)
codec = cap.get(cv2.CAP_PROP_FOURCC)
print( codec)
count = 0
fastNum= 0
countlast = 0
probablyNum = 0
thisFast = 0
directory = "C:/Users/yg155d/Documents/img/Brotje/"
fasttemp = cv2.imread(directory + "Fastener.png")
holetemp = cv2.imread(directory + "TemplateHole.png")
method = eval('cv2.TM_CCOEFF_NORMED')
z = load_model( "E:/img/Animals/Models/ConvNet_CIFAR10_DataAugModel100b.dnn")
skip = 1
fail = 0
#if not os.path.exists("C:/Users/yg155d/Documents/img/Brotje/RawImages1/"):
#    os.makedirs("C:/Users/yg155d/Documents/img/Brotje/RawImages1/")
#    os.makedirs("C:/Users/yg155d/Documents/img/Brotje/RawImages1/ExtraImages/")
#    os.makedirs("C:/Users/yg155d/Documents/img/Brotje/RawImages1/IndFastener/")
#    os.makedirs("C:/Users/yg155d/Documents/img/Brotje/RawImages1/possible/")

for video in Videos:
    print(video)
    cap = cv2.VideoCapture(0)
    inVideo = True
    while(inVideo):
    
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret: #if frame back ok, proceed to process
            fail = 0
            # Our operations on the frame come here
        
            count += 1
            # Display the resulting frame
            if count % skip == 0:

                #frameCV = frame[1:288,1:353] #crop image to left corner frame
                cv2.imshow('frame',frame)
                #add in logic to filter out only the images with completed fasteners.  #maybe change number of clicks between screenshots depending on current image.  That way we process more images near when a fastener is complete.
                ##########comment out section to run regular code
                ###This section will use neural net to classify if in position once trained model created.
                frame1 = Image.fromarray(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                pic = flattenImages(100,100,frame1)
                #frame.resize((60,50),Image.ANTIALIAS)
                #rgb_image = np.asarray(frame, dtype=np.float32)
                #bgr_image = rgb_image[..., [2, 1, 0]]
                #pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
                predictions = np.squeeze(z.eval({z.arguments[0]:[pic]}))
                top_class = np.argmax(predictions)
                
                print("topClass",top_class, Paths[top_class], predictions[top_class]);
                #thisScore = predictions[0]
                #if top_class == 0 and thisScore >5.2:
                #    res = cv2.matchTemplate(frameCV,fasttemp,method)
                #    threshold = 0.54
                #    loc = np.where( res >= threshold)
                #    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                #    if max_val>threshold:
                        
                #        if newfastNum > fastNum:# been a while since we saw a fastener, must be a new one
                #            if os.path.exists("C:/Users/yg155d/Documents/img/Brotje/RawImages1/IndFastener/Fasteners{}{}".format(video,fastNum)+".png"):
                #                os.rename("C:/Users/yg155d/Documents/img/Brotje/RawImages1/IndFastener/Fasteners{}{}".format(video,fastNum)+".png","C:/Users/yg155d/Documents/img/Brotje/RawImages1/IndFastener/Fasteners{}{}-{}".format(video,fastNum,thisFast)+".png")
                #            fastNum = fastNum+1
                #            thisFast = 0
                #            highestScore = 0
                #        else:
                #            thisFast = thisFast+1
                #        print ("confirmed fastener", fastNum,thisFast)    
                #        if thisFast<20:
                #            frame1 = frameCV[(max_loc[0]+10):(max_loc[0]+115),(max_loc[1]-40):(max_loc[1]+130)]
                #            if thisScore > highestScore:
                #                highestScore = thisScore
                #                imageFileName = "C:/Users/yg155d/Documents/img/Brotje/RawImages1/IndFastener/Fasteners{}{}".format(video,fastNum)
                #                cv2.imwrite(imageFileName+".png",frame1)
                            
                #            imageFileName = "C:/Users/yg155d/Documents/img/Brotje/RawImages1/Fasteners{}{}_{}".format(video,fastNum,thisFast)
                #            cv2.imwrite(imageFileName+".png",frame1)
                #            #cv2.imwrite(imageFileName+"L.png",frame)
                #        else:
                #            frame1 = frameCV[(max_loc[0]+10):(max_loc[0]+115),(max_loc[1]-40):(max_loc[1]+130)]
                #            imageFileName = "C:/Users/yg155d/Documents/img/Brotje/RawImages1/ExtraImages/Fasteners{}{}_{}".format(video,fastNum,thisFast)
                #            cv2.imwrite(imageFileName+".png",frame1)
                #        countlast = count
                #        skip = 2
                        
                            
                #    else:
                #        print("FailTemplate, MaxVal = ", max_val)
                #        #cv2.imshow('frame',frameCV)
                #        probablyNum = probablyNum+1
                #        imageFileName = "C:/Users/yg155d/Documents/img/Brotje/RawImages1/possible/FailTemplate{}{}_{}".format(video,fastNum,probablyNum)
                #        frameCV = cv2.resize(frameCV,(int(353/4),int(288/4)),interpolation = cv2.INTER_AREA)
                #        cv2.imwrite(imageFileName+".png",frameCV)
                #        skip = 1
                #elif predictions[0] >3.5:
                #    print("FailNN, Top Pred ", top_class, predictions[0])
                #    #cv2.imshow('frame',frameCV)
                #    probablyNum = probablyNum+1
                #    imageFileName = "C:/Users/yg155d/Documents/img/Brotje/RawImages1/possible/FailNN{}{}_{}-{:1.2f}-{}".format(video,fastNum,probablyNum,predictions[0], top_class)
                #    frameCV = cv2.resize(frameCV,(int(353/4),int(288/4)),interpolation = cv2.INTER_AREA)
                #    cv2.imwrite(imageFileName+".png",frameCV)
                #    skip = 2
                #elif top_class == 1:
                #    skip = 4
                #    newfastNum = fastNum+1
                #    print("see hole",predictions)
                #elif top_class == 2:
                #    skip = 3
                #    newfastNum = fastNum+1
                #    print("will see fastener soon")
                #elif top_class == 3:
                #    skip = 15
                #    print("aint seen nothin")
                    


                   
                    
                    #print("nothing to see here")
        elif fail >20:
            inVideo = False
        else:
            fail = fail+1  
    cap.release()
    
                



            #frame = Image.fromarray(frame)
        
            #rectangle = (1,1,352,288)
            #frame = frame.crop(rectangle)
            #pic = flattenImages(108,132,frame)
            #imageFileName = "C:/Users/yg155d/Documents/img/Brotje/RawImages/img{}".format(count)
            #frame.save(imageFileName+ ".png")

            #predictions = np.squeeze(z.eval(pic))
            #top_class = np.argmax(predictions)
            #if predictions[0] > 1 and predictions[1] >-.9  :
            #    skip = 5
            #    imageFileName = "C:/Users/yg155d/Documents/img/Brotje/RawImages/img{}_{}".format(count,predictions[2])
            #    #frame.save(imageFileName+ ".png")
            #    print ("See Fastener",predictions)

                
            #else:
            #    skip = 5
            #    print ("No fastener", predictions)

            ####Basic computer vision section
            #im = cv2.imread(imageFileName+".bmp")
            #gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            ##cv2.imshow("gray",gray)
            ## detect circles in the image
            #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.1, 80)
            ## ensure at least some circles were found
            #if circles is not None:
	           # # convert the (x, y) coordinates and radius of the circles to integers
            #    circles = np.round(circles[0, :]).astype("int")
 
	           # # loop over the (x, y) coordinates and radius of the circles
            #    for (x, y, r) in circles:
		          #  # draw the circle in the output image, then draw a rectangle
		          #  # corresponding to the center of the circle
            #        cv2.circle(im, (x, y), r, (0, 255, 0), 4)
            #        cv2.rectangle(im, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
	           # # show the output image
            #    #cv2.imshow("output", im)
            #    cv2.imwrite(imageFileName + ".png",im)
            #    #cv2.waitKey(0)
            #else: 
            #    cv2.imwrite(imageFileName+"no.png",im)

    
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()