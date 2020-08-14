####### Author      #########
#Created by Clinton Campbell, yg155d
#Property of the Boeing Company
#27 March 2017
####### Description #########
#This script manipulates the mouse and takes screenshots to grab specific images of installed fasteners for Brotje SC.
#This is needed to grab images from proprietary format video to create a training set for machine learning.
####### Istructions #########
# Start MiniPlayer  and open the video file to capture images from.
# Select "Window Mode": 1 Channel.  Scroll to the correct channel to take images from. Move the video to where you want to start from.
# Run the script. Adjust values noted below to center image correctly and click on next frame button correctly. 
####### TO DO       #########
#set up computer vision that will only save the images that have clear fastener views. Perhaps create neural network that detects these views.
#create tool that displays pixel value from mouse click to easily reconfigure


import numpy as np
import cv2
import time
import win32api, win32con,win32gui,win32ui
import re
from PIL import Image
import os
from Utilities import flattenImages
from Utilities import resizeImage
FileLocation = "C:/Users/yg155d/Documents/img/Brotje/images/"
def click(x,y):
    x1,y1 = win32api.GetCursorPos()
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    win32api.SetCursorPos((x1,y1)) #put the mouse back for less annoyance

def image_grab_win32(w,left,top,right,bot,bmpfilenamename):  #faster method to pull a screen shot
    wid = right - left
    h = bot - top
    wDC = win32gui.GetWindowDC(w._handle)
    dcObj=win32ui.CreateDCFromHandle(wDC)
    cDC=dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, wid, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0),(wid, h) , dcObj, (left,top), win32con.SRCCOPY)
    dataBitMap.SaveBitmapFile(cDC, bmpfilenamename+".bmp")
    #release objects
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(w._handle, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    
class WindowMgr: #this section helps find the window we want to pull to the front
    def __init__ (self):
            """Constructor"""
            self._handle = None

    def find_window(self, class_name, window_name = None):
        """find a window by its class_name"""
        self._handle = win32gui.FindWindow(class_name, window_name)

    def _window_enum_callback(self, hwnd, wildcard):
        '''Pass to win32gui.EnumWindows() to check all the opened windows'''
        if re.match(wildcard, str(win32gui.GetWindowText(hwnd))) != None:
            self._handle = hwnd

    def find_window_wildcard(self, wildcard):
        self._handle = None
        win32gui.EnumWindows(self._window_enum_callback, wildcard)

    def set_foreground(self):
        """put the window in the foreground"""
        win32gui.SetForegroundWindow(self._handle)


if __name__ == "__main__":
    i = 0
    #click(1000,1000)
    w=WindowMgr()
    name = w.find_window_wildcard(".*MiniPlayer*.") #find the miniplayer screen
    w.set_foreground()
    #may need to change this to size of window on your monitor. 
    win32gui.MoveWindow(w._handle, 0, 0, 1250, 1033, True)#reshape the window to the normal aspect ratio.  Full screen on my monitor stretches image. Should be approximately what stretching to the top does.
    #z = load_model( "C:/Users/yg155d/Documents/img/Brotje/Models/atLocationDiscerner.model.dnn")
    for i in range(1000):
        w.set_foreground()#reframe on miniplayer
        m = 0
        for m in range(2):#click through several times to skip m images
            click(130,980) #click where the next frame button is.  Adjust as necessary
        time.sleep(1/20) #1/25 is the frame rate for this video
        #change pixels from top left corner of main screen to match area of screen shot
        imageFileName = filelocation + "img{}".format(i) #index each image
        image_grab_win32(w,6,67,622,505,imageFileName)#take screen shot of box,and save
        im= Image.open(imageFileName+ ".bmp")
        im.save(imageFileName + ".png")
        
        
        
        #add in logic to filter out only the images with completed fasteners.  #maybe change number of clicks between screenshots depending on current image.  That way we process more images near when a fastener is complete.
        ##########comment out section to run regular code
        ###This section will use neural net to classify if in position once trained model created.
        #pic = flattenImages(im)
        #evalu = z.eval(pic)
        ##print (evalu)
        #predictions = np.squeeze(z.eval(pic))
        #top_class = np.argmax(predictions)


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


        ###########
        #Continue regular code.
        os.remove(imageFileName+".bmp")

