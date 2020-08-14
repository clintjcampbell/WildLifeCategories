import numpy as np
import cv2
import time
from Utilities import flattenImages
from PIL import Image
import cntk
import _cntk_py
from cntk.ops.functions import load_model
Videos = ["ch2_223_0-140.avi","ch2_223_140_320.avi","ch2_223_320_500.avi","ch2_223_5-7.avi","ch2_223_7-9.avi","ch2_223_9-11.avi","ch2_223_11-13.avi","ch2_223_13-15.avi","ch2_223_15-17.avi","ch2_223_17-19.avi","ch2_223_19-21.avi","ch2_223_21-23.avi","ch2_223_23-01.avi","ch2_224_01-03.avi","ch2_224_03-05.avi","ch2_224_05-07.avi","ch2_224_07-09.avi","ch2_224_09-11.avi","ch2_224_11-13.avi","ch2_224_13-15.avi","ch2_224_15-17.avi","ch2_224_17-19.avi","ch2_224_19-21.avi","ch2_224_21-23.5.avi"]
cap = cv2.VideoCapture("C:/Users/yg155d/Documents/img/Brotje/Videos/Ch2_223_0-140.avi")
codec = cap.get(cv2.CAP_PROP_FOURCC)
print( codec)
count = 0
fastNum= 0
countlast = 0
thisFast = 0
directory = "C:/Users/yg155d/Documents/img/Brotje/"
fasttemp = cv2.imread(directory + "Fastener.png")
holetemp = cv2.imread(directory + "TemplateHole.png")
method = eval('cv2.TM_CCOEFF_NORMED')
#z = load_model( "C:/Users/yg155d/Documents/img/Brotje/Models/atLocationDiscerner2c.model.dnn")
skip = 1
fail = 0
for video in Videos:
    print(video)
    cap = cv2.VideoCapture("C:/Users/yg155d/Documents/img/Brotje/Videos/"+video)
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
            
                
                frame = frame[1:288,1:353] #crop image to left corner frame
                #cv2.imshow('frame',frame)
                #add in logic to filter out only the images with completed fasteners.  #maybe change number of clicks between screenshots depending on current image.  That way we process more images near when a fastener is complete.
                ##########comment out section to run regular code
                ###This section will use neural net to classify if in position once trained model created.
                res = cv2.matchTemplate(frame,fasttemp,method)
                threshold = 0.75
                loc = np.where( res >= threshold)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val >threshold:
                    skip = 1;
                    #print("think Saw Fastener",max_val)
                    res2 = cv2.matchTemplate(frame,holetemp,method)
                    threshold = 0.85
                    min_val, max_val, min_loc, max_loch = cv2.minMaxLoc(res2)
                    if max_val <threshold:
                        if count - countlast >3:# been a while since we saw a fastener, must be a new one
                            fastNum = fastNum+1
                            thisFast = 0
                        else:
                            thisFast = thisFast+1
                        print ("confirmed fastener", fastNum,thisFast)
                        
                        if thisFast<15:
                            frame1 = frame[(max_loc[0]+10):(max_loc[0]+115),(max_loc[1]-40):(max_loc[1]+130)]
                            imageFileName = "C:/Users/yg155d/Documents/img/Brotje/RawImages/Fastener{}{}_{}".format(video,fastNum,thisFast)
                            cv2.imwrite(imageFileName+".png",frame1)
                            cv2.imwrite(imageFileName+"L.png",frame)
                        countlast = count
                        skip = 1
                        if thisFast == 5:
                            #frame = frame[(max_loc[0]+10):(max_loc[0]+100),(max_loc[1]-50):(max_loc[1]+110)]
                            imageFileName = "C:/Users/yg155d/Documents/img/Brotje/RawImages/IndFastener/Fastener{}{}".format(video,fastNum)
                            cv2.imwrite(imageFileName+".png",frame1)

                    else:
                        print ("saw a hole",max_val)
                        frame1 = frame[(max_loc[0]+10):(max_loc[0]+115),(max_loc[1]-40):(max_loc[1]+130)]
                        imageFileName = "C:/Users/yg155d/Documents/img/Brotje/RawImages/Holes/Hole{}{}".format(video,count)
                        cv2.imwrite(imageFileName+".png",frame1)
                        cv2.imwrite(imageFileName+"L.png",frame)
                        
                else:
                    skip =10
                    imageFileName = "C:/Users/yg155d/Documents/img/Brotje/RawImages/NoFastener/None{}{}".format(video,count)
                    cv2.imwrite(imageFileName+".png",frame)
                    
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

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()