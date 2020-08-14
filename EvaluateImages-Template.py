import numpy as np
import cv2
import time
from Utilities import flattenImages, createImageList
from PIL import Image
import cntk
import os
import _cntk_py
from cntk.ops.functions import load_model
directory  = "/"
Paths = [os.path.join(directory,'2'), os.path.join(directory,'')]
OutIm,Outcat = createImageList(Paths)
count = 0
#z = load_model( "atLocationDiscernerRes1.model.dnn")
ZeroCorrect = 0
ZeroWrong = 0
OneCorrect = 0
OneWrong = 0
zeroPredict = []
onePredict = []
index = 50000;
template = cv2.imread(directory + "Fastener.png")
while index < len(OutIm):
    # Capture frame-by-frame
    frame = cv2.imread(OutIm[index])
    method = eval('cv2.TM_CCOEFF_NORMED')

    #frame = Image.open(OutIm[index])   
    res = cv2.matchTemplate(frame,template,method)
    threshold = 0.8
    loc = np.where( res >= threshold)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if Outcat[index] == 1 and max_val >threshold:
        OneCorrect=+1
        print(Outcat[index],"correct show",max_val)
        #cv2.imshow(frame)
    elif Outcat[index] == 1 and max_val <threshold:
        OneWrong=+1
        print(Outcat[index],"should have shown",max_val,OutIm[index])
        
    elif Outcat[index]==0 and max_val > threshold:
        print (Outcat[index],"false show",max_val,OutIm[index])   
        ZeroWrong=+1 

    else:
        print ("Correct False", max_val)
        ZeroCorrect+=1
    index+=3
print (OneCorrect)
print (OneWrong)
print(ZeroCorrect)
print (ZeroWrong)

    #cv2.imshow('frame',frame)
    #Image._show(frame)
#    pic = flattenImages(50,60,frame)
#    #imageFileName = "img{}".format(count)
#    #frame.save(imageFileName+ ".png")

#    predictions = np.squeeze(z.eval(pic))
#    top_class = np.argmax(predictions)
#    print (index, "Category: ",Outcat[index], "Prediction: ",top_class, predictions)
#    if Outcat[index] == 0:
#        zeroPredict.append(predictions)
#    if Outcat[index] ==1: 
#        onePredict.append(predictions)
#    if top_class == Outcat[index]:
#        if top_class == 0:
#            ZeroCorrect +=1
#        else:
#            OneCorrect +=1
#    elif  top_class == 1:
#        ZeroWrong +=1
#    else: 
#        OneWrong +=1
#    if Outcat[index] == 0:
#        index += 100
#    else: index += 2
#print ("Final Report out of accuracy on images shown:")
#print ("ZeroCorrect: ", ZeroCorrect)
#print("OneCorrect: ", OneCorrect)
#print("ZeroWrong: ", ZeroWrong)
#print("OneWrong: ", OneWrong)
#print("zerozeroMax",zeroPredict[0][:].max())
#print("zerozeromin",zeroPredict[0][:].min())
#print("zerozeromean",zeroPredict[0][:].mean())
#print("zeroonemax",zeroPredict[1][:].max())
#print("zeroonemin",zeroPredict[1][:].min())
#print("zeroonemean",zeroPredict[1][:].mean())
#print("onezeromax",onePredict[0][:].max())
#print("onezeromin",onePredict[0][:].min())
#print("onezeromean",onePredict[0][:].mean())
#print("oneonemax",onePredict[1][:].max())
#print("oneonemin",onePredict[1][:].min())
#print("oneonemean",onePredict[1][:].mean())

    #if predictions[1] >1  :
    #    skip = 5
    #    imageFileName = "img{}_{}".format(count,predictions[2])
    #    #frame.save(imageFileName+ ".png")
    #    print ("See Fastener",predictions)

                
    #elif predictions[0] > 1:
    #    skip = 5
    #    print ("installing fastener", predictions)
    #else: 
    #    skip = 5
    #    print("no fasteners", predictions)
    #time.sleep(.02)

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

    
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# When everything done, release the capture

    #rectangle = (53,48,297,249)
    #frame = frame.crop(rectangle)  
    #location= str(OutIm[index])
    #location = location.replace(".png","crop.png")
    #frame.save(location)
