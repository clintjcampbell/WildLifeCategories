# -*- coding: utf-8 -*-
#!/usr/bin/python
#!/usr/bin/python import ConvNet_Animals_DataAugCurrent
"""
Created on Fri Mar 31 09:40:13 2017

@author: yg155d
"""
#Maybe change name to data utils and bring other tools into this?
import os
from os import listdir
import cv2
import time
from os.path import isfile, join
from PIL import Image
import numpy as np
from scipy.misc import imread
from cntk.initializer import he_normal
from cntk.layers import AveragePooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu
import ConvNet_Animals_DataAugCurrent

###take in list of images in different folder.  return list of images with coresponding category numbers
def createImageList(Paths):
    print ('starting')
    #f = open(output,'w')
    category = 0
    
    OutputImages =[]
    OutputCategories = []
    for mypath in Paths:   
        print ('path:' + mypath)
        onlyfiles = [fl for fl in listdir(mypath) if fl.endswith('.png') or fl.endswith('.bmp')]#filters for only files that we want
        length = len(onlyfiles)
        outLength = len(OutputImages)
        for count in range(length):
            OutputImages.append(mypath+ "/" + onlyfiles[count])
            OutputCategories.append(category)
        category+=1

    print ('done creating mapping file')
 
    return OutputImages,OutputCategories

def CreateMapFile(directory,output, Paths,maxLength):
    
    print ('starting')
    f = open(output,'w')
    category = 0
    count = 0
    
    for mypath in Paths:   
        print ('path:' + mypath)
        onlyfiles = [fl for fl in listdir(mypath) if fl.endswith('.png') or fl.endswith('.bmp')]#filters for only files that we want
        length = len(onlyfiles)
        thisCount = 0
        repeat =1
        if length <maxLength:
            repeat = int(maxLength/length)+1;
        for fl2 in onlyfiles:
            for rep in range(repeat):
                filename = mypath + "/" + fl2
                f.write(filename + "\t")
                f.write(str(category))
                f.write("\n")
                count+=1
                thisCount+=1
        print("Number in Category", thisCount)
        category+=1

    print ('done creating mapping file')
    f.close() 
    print ("files", count)
    return count
def flattenImagesBW(imDimWd,imDimTl, img):
    img = img.convert('L')
    img = img.resize((imDimWd,imDimTl),Image.ANTIALIAS)
    img = np.array(img.getdata()).reshape([1,3,img.size[0],img.size[1]])
    return img
def flattenImages(imDimWd, imDimTl, img ):
    img = img.resize((imDimWd,imDimTl),Image.ANTIALIAS)
    rgb_image = np.asarray(img, dtype=np.float32)
    bgr_image = rgb_image[..., [2, 1, 0]]
    img = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
    return img
def resizeImage(imgPath):
    #img = Image.open(imgPath)
    img = img.resize((imDimWd,imDimTl),Image.ANTIALIAS)
    return img





#
# Resnet building blocks
#
def conv_bn(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False)(c)
    return r

def conv_bn_relu(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    r = conv_bn(input, filter_size, num_filters, strides, init) 
    return relu(r)

def resnet_basic(input, num_filters):
    c1 = conv_bn_relu(input, (3,3), num_filters)
    c2 = conv_bn(c1, (3,3), num_filters)
    p  = c2 + input
    return relu(p)

def resnet_basic_inc(input, num_filters, strides=(2,2)):
    c1 = conv_bn_relu(input, (3,3), num_filters, strides)
    c2 = conv_bn(c1, (3,3), num_filters)
    s  = conv_bn(input, (1,1), num_filters, strides)
    p  = c2 + s
    return relu(p)

def resnet_basic_stack(input, num_stack_layers, num_filters): 
    assert (num_stack_layers >= 0)
    l = input 
    for _ in range(num_stack_layers): 
        l = resnet_basic(l, num_filters)
    return l 

#   
# Defines the residual network model for classifying images
#
def create_resnet_model(input, num_stack_layers, num_classes):
    c_map = [16, 32, 64]

    conv = conv_bn_relu(input, (3,3), c_map[0])
    r1 = resnet_basic_stack(conv, num_stack_layers, c_map[0])

    r2_1 = resnet_basic_inc(r1, c_map[1])
    r2_2 = resnet_basic_stack(r2_1, num_stack_layers-1, c_map[1])

    r3_1 = resnet_basic_inc(r2_2, c_map[2])
    r3_2 = resnet_basic_stack(r3_1, num_stack_layers-1, c_map[2])

    # Global average pooling and output
    pool = AveragePooling(filter_shape=(8,8))(r3_2) 
    z = Dense(num_classes)(pool)
    return z

def findInputSizeNN(z):
    zSize = str(z.inputs[len(z.inputs[:])-1]).split(",")
    zSize = zSize[3].replace("[","").replace("]","").replace(")","").split("x")
    zSizew = int(zSize[2])
    zSizet = int(zSize[1])
    return zSizew, zSizet
def findImageLocationNN(pic1,z, name, index):
    #add in logic to scroll over image and analyze sections and return high score locations similar to template
    #Try to reference center or top left for scoring?
    NNw,NNt = findInputSizeNN(z)
    pic =np.asarray(pic1)
    picw=pic1.size[0]
    pict = pic1.size[1]
    windx = min(picw,pict)-3
    windy = min(picw,pict)-3
    slide = 3
    i = int(0/slide)
    j = int(0/slide)
    #img = np.array(pic)
    results=[ [0]*(picw-windx) for _ in range(pict-windy) ]
    MaxVal = -25
    x = 0
    y = 0        
    x1= x
    y1 = y
    widey = windy
    widex = windx
    shrink = 5
    count = 0 
    
    
    MaxVal = -5000
    while windx >40 and windy >40:
        endi = (picw-windx+shrink)
        endj = (pict -windy+shrink)
        while (i *slide) < (endi):
            
            while (j * slide) <(endj):
                #img3 = img
                #do some stuff on sliding window and evaluating
                #do some stuff on resizing the window as well in general, but probably same size window for fasteners.
                #rect = (slide * i,slide *j,slide*i + windx,slide*j + windy)
                picWindow =pic1.crop(((slide*i), (slide*j), (slide*i + windx), (slide*j + windy)))
                #picCV = np.asarray(picWindow)
                #cv2.imshow("a",picCV)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
                #time.sleep(.5)
                pic2 = flattenImages(NNw,NNt,picWindow)
                predictions = np.squeeze(z.eval({z.arguments[0]:[pic2]}))
                #top_class = np.argmax(predictions)
                if predictions[index] > MaxVal:
                    MaxVal = predictions[index]
                    x1 = i*slide
                    y1 = j *slide
                    widex = windx
                    widey = windy
                    
                #count = count+1
                #if predictions[0] > 2.5:
                #    picWindow.save("C:/Users/yg155d/Documents/img/Brotje/ReviewImages/VisFastener/Fascrop" + str(name) + "_" + str(j) +"-" + str(i) + " " + str(MaxVal) + ".png")
                #else:
                #    picWindow.save("C:/Users/yg155d/Documents/img/Brotje/ReviewImages/NoFastener/NoFascrop" + str(name) + "_" + str(j) +"-" + str(i) + " " + str(MaxVal) + ".png")
                
                j = j+1
            j = 1
            if y1>11:
                j = y1-5
            i = i+1
        i = 1
        if x1>11:
            i = x1-5
        windx = windx-7
        windy = windy-7
    #if slide == 1:
    #    slide = 0
    #else:
    #    #slide -= 1
    #    i = int((x1-int(slide/2)-1)/slide)
    #    j = int((y1-int(slide/2)-1)/slide)
        
        
    pixx =i*slide
    pixy = j*slide
    endi = i+2*slide
    endj = j+2*slide
    #maxrow = max(results)
    #max1 = max(maxrow)
    #y = maxrow.index(max1)
    #x = results.index(maxrow)
    #if MaxVal > 0 and MaxVal < 3.1:
    #    picWindow.save("C:/Users/yg155d/Documents/img/Brotje/ReviewImages/BadFastenerView/Fascrop" + str(name) + "_" + str(j) +"-" + str(i) + " " + str(MaxVal) + ".png")
    #elif MaxVal <0:
    #    picWindow.save("C:/Users/yg155d/Documents/img/Brotje/ReviewImages/DrillingImages/Fascrop" + str(name) + "_" + str(j) +"-" + str(i) + " " + str(MaxVal) + ".png")
    #else: 
    #    picWindow.save("C:/Users/yg155d/Documents/img/Brotje/ReviewImages/VisFastener2/Fascrop" + str(name) + "_" + str(j) +"-" + str(i) + " " + str(MaxVal) + ".png")
    print("animal at ",x1,y1, widex,widey, MaxVal)
    #resultIm = pic.crop((x1,y1,x1+widex,y1+widey))
    #origIm = np.array(pic)
    #resultIm2= np.array(resultIm)
    #cv2.imshow("a",resultIm2)
    #cv2.imshow("original",origIm)
    
    return x1, y1,widex,widey, MaxVal


if __name__=='__main__':
    #directory  = "C:/Users/yg155d/Documents/img/Brotje/" 
    while True:
        os.system('python ConvNet_Animals_DataAugCurrent.py')
    #output = os.path.join(directory,'BrotjeCategoryMapping.txt')
    #CreateMapFile(directory,output)
    #Paths = [os.path.join(directory,'NoFastener2'), os.path.join(directory,'VisFastener')]
    #OutIm,Outcat = createImageList(Paths)
    #print(Outcat[2000],OutIm[2000])
    