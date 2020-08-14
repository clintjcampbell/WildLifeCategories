import numpy as np
import cv2
import time
from Utilities import flattenImages, createImageList
from PIL import Image
import cntk
import os
import _cntk_py
from cntk.ops.functions import load_model
directory  = "C:/Users/yg155d/Documents/img/Brotje/"
Paths = [os.path.join(directory,'VisFastener'), os.path.join(directory,'HolesVisible'),os.path.join(directory,'DrillingImages'),os.path.join(directory,'NoFastener')]
newPaths = ["/Brotje/ReviewImages/VisFastener/","/Brotje/ReviewImages/HolesVisible/","/Brotje/ReviewImages/DrillingImages/","/Brotje/ReviewImages/NoFastener/"]
#Paths = [os.path.join(directory,'Test1')]
OutIm,Outcat = createImageList(Paths)
count = 0
z = load_model( "C:/Users/yg155d/Documents/img/Brotje/Models/atLocationDiscerner2c.model.dnn")
#size = z.__getattribute__(input)
ZeroCorrect = 0
ZeroWrong = 0
OneCorrect = 0
OneWrong = 0
zeroPredict = []
onePredict = []
zeroPredictWrong = []
onePredictWrong = []
index = 13884;
template = cv2.imread(directory + "Fastener.png")
while index < len(OutIm):
    frame = Image.open(OutIm[index]) 
    frame = frame.resize((60,50),Image.ANTIALIAS)
    rgb_image = np.asarray(frame, dtype=np.float32)
    bgr_image = rgb_image[..., [2, 1, 0]]
    pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))



    #pic = flattenImages(50,60,frame)
#    #imageFileName = "C:/Users/yg155d/Documents/img/Brotje/RawImages/img{}".format(count)
#    #frame.save(imageFileName+ ".png")
    

    predictions = np.squeeze(z.eval({z.arguments[0]:[pic]}))
    top_class = np.argmax(predictions)
    print (index, "Category: ",Outcat[index], "Prediction: ",top_class, predictions)
    if predictions[Outcat[index]] < 2:
        newLoc = OutIm[index].replace("/Brotje/","/Brotje/ReviewImages/")
        os.rename(OutIm[index],newLoc)
        print ("moved to ", newLoc)
    index = index+1
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

#print("oneonemax",onePredictWrong[1][:].max())
#print("oneonemin",onePredictWrong[1][:].min())
#print("oneonemean",onePredictWrong[1][:].mean())
#print("zeroonemax",zeroPredictWrong[1][:].max())
#print("oneonemin",zeroPredictWrong[1][:].min())
#print("oneonemean",zeroPredictWrong[1][:].mean())
    #if predictions[1] >1  :
    #    skip = 5
    #    imageFileName = "C:/Users/yg155d/Documents/img/Brotje/RawImages/img{}_{}".format(count,predictions[2])
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