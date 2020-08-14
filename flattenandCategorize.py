# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:40:13 2016

@author: Clinton Campbell
"""
from PIL import Image
#from cntk.ops.functions import load_model
import numpy as np
from scipy.misc import imread
import os
from os import listdir
from os.path import isfile, join
import xml.etree.cElementTree as et
import xml.dom.minidom
#import cntk as ct
directory = 'E:/img/Animals/'
output = directory+'AnimalsFullSize.txt'
imDimWd = 200
imDimTl = 200
def resizeImage(imgPath,savePath):
    img = Image.open(imgPath)
    img = img.resize((imDimWd,imDimTl),Image.ANTIALIAS)
    img.save(savePath)
def flattenImages(imgPath, savePath):
    img = Image.open(imgPath)
    img = img.resize((imDimWd,imDimTl),Image.ANTIALIAS)
    #img.save(imgPath2)
#    f1 = im2.getdata()
    imList = list(img.getdata())
    #mode = img.mode    
    #sizeIm = img.size
    rlist = []
    glist = []
    blist = []
    count = len(imList)
    for i in range(0,count):
    #for List in imList:
    #    for i in List:
        r,g,b = imList[i]
        rlist.append(r)
        glist.append(g)
        blist.append(b)
        
    imgVec = rlist + glist + blist
    return imgVec
    
def flattenImagesBW(imgPath,imgPath2):
    img = Image.open(imgPath)
    img = img.convert('L')
    img = img.resize((imDimWd,imDimTl),Image.ANTIALIAS)
    #img.save(imgPath2)
#    f1 = im2.getdata()
    imList = list(img.getdata())
#    #mode = img.mode    
#    #sizeIm = img.size
#    rlist = []
#    glist = []
    blist = []
    count = len(imList)
    for i in range(0,count):
#    #for List in imList:
#    #    for i in List:
        b = imList[i]
#        #rlist.append(r)
#        #glist.append(g)
        blist.append(b)
#       
#    imgVec = blist 
    return blist
def flattenImagesBW2(imgPath,imgPath2):
    img = Image.open(imgPath)
   
    img = img.resize((imDimWd,imDimTl),Image.ANTIALIAS)
    #img.save(imgPath2)
#    f1 = im2.getdata()
    imList = list(img.getdata())
#    #mode = img.mode    
#    #sizeIm = img.size
#    rlist = []
#    glist = []
    blist = []
    count = len(imList)
    for i in range(0,count):
#    #for List in imList:
#    #    for i in List:
        r,g,b = imList[i]
        d = 0
        
        #Background
        if ((r > 250) & (r < 256)) & ((g >250) & (g <256)) & ((b>250) & (b< 256)) :
            d = 0


        #post
        if ((r > 233) & (r < 255)) & ((g >60) & (g <70)) & ((b>0) & (b< 55)) :
            d = 100
        #clip
        if ((r > 250) & (r <= 255)) & ((g >250) & (g <=255)) & ((b>=0) & (b< 10)) :
            d = 130
        #Rack
        if ((r > 250) & (r <= 255)) & ((g >=0) & (g <20)) & ((b>=0) & (b< 20)) :
            d = 100
        #ball
        if ((r >= 0) & (r < 15)) & ((g >250) & (g <=255)) & ((b>=0) & (b< 10)) :
            d = 255
        #stringer
        if ((r >= 0) & (r < 10)) & ((g >250) & (g <=255)) & ((b>250) & (b<= 255)) :
            d = 200
#        #rlist.append(r)
#        #glist.append(g)
        blist.append(d)
#       
#    imgVec = blist 
    return blist

def unflattenImages(imgVec, sizex, sizey):
    size1 =(len(imgVec)+1)/3
    imList = []
    for i in range(0,size1-1):
        r = imgVec[i]
        g = imgVec[i+size1]
        b = imgVec[i+2*size1]
        imList.append((r,g,b))
    img = Image.new('RGB',(sizex,sizey))
    img.putdata(imList)

    return img
def unflattenImagesBW(imgVec, sizex, sizey):
    size1 =(len(imgVec))
    imList = []
    for i in range(0,size1):
        r = imgVec[i]
        imList.append(r)
    img = Image.new('l',(sizex,sizey))
    img.putdata(imList)

    return img
def saveTrainImages():

    dataMean = np.zeros((3, imDimWd, imDimTl)) # mean is in CHW format.
    dataMean = dataMean / (50 * 1000)
    saveMean('CIFAR-10_mean.xml', dataMean)
def saveMean(fname, data):
    root = et.Element('opencv_storage')
    et.SubElement(root, 'Channel').text = '3'
    et.SubElement(root, 'Row').text = str(imDimTl)
    et.SubElement(root, 'Col').text = str(imDimWd)
    meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
    et.SubElement(meanImg, 'rows').text = '1'
    et.SubElement(meanImg, 'cols').text = str(imDimWd * imDimTl * 3)
    et.SubElement(meanImg, 'dt').text = 'f'
    et.SubElement(meanImg, 'data').text = ' '.join(['%e' % n for n in np.reshape(data, (imDimWd * imDimTl * 3))])

    tree = et.ElementTree(root)
    tree.write(fname)
    x = xml.dom.minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent = '  '))
def indexLabels(ind, count):
    label = []
    for i in range(count):
        if i == ind:
            label.append(1)
        else: 
            label.append(0)
    return label


def writeCategories(output, Paths):
    print ('starting')
    f = open(output,'w+')
    
    ind = 0
    j = 0
    for mypath in Paths:  
        print ('path:' + mypath)
        onlyfiles = [fl for fl in listdir(mypath) if fl.endswith('.jpeg')]
        length = len(onlyfiles)
        i = 0
        repeat = 0
        if length!= 0:
            repeat = int(5000/length)
        if repeat == 0: repeat = 1
    
        for fl2 in onlyfiles:
            #decide = fl2.split('_') 
            #fl1 = fl2.replace('2.bmp','1.bmp')
            #length2 = fl2.index(".") +2       
            #manipulate path to pull 2 images and turn into label and features.
            #flName = fl2[:length2]
            #ind = partName.index(flName)
    #        label = indexLabels(ind,25)
    #        labels = "%s" %label
    #        labels = labels.replace(',', '')
    #        labels = labels.replace('[', '')
    #        labels = labels.replace(']', '')
            #pth1 =mypath+fl1
            pth2 =mypath+fl2
    #        img_vec1 = flattenImages(pth2)
    #        features = "%s" %img_vec1
    #        features = features.replace(',', '')
    #        features = features.replace('[', '')
    #        features = features.replace(']', '')
            #img_vec2 = flattenImagesBW(pth2,directory + 'S4b/'+fl2)
            #unflat = unflattenImagesBW(img_vec2,100,100)
            #unflat.save(directory + 'Sout/'+fl2)
            #outputs = "%s" %img_vec2
            #outputs = outputs.replace(',', '')
            #outputs = outputs.replace('[', '')
            #outputs = outputs.replace(']', '')
    #        f.write('|labels ')
    #        f.writelines(labels)
            #resizeImage(pth2,directory + "Animals2/" + fl2)
            for rep in range(repeat):
                f.write(mypath + fl2 +'\t{}'.format(ind))
                j = j+1
            #f.write(pth2 +'\t{}'.format(ind))
            #f.write('|outputs ') # python will convert \n to os.linesep
            #f.writelines(outputs)
    #        f.write(' |features ')
    #        f.writelines(features)
                f.write('\n')
        
            if (i%500 == 0):
                percent = (i / (repeat*length))*100
                print ('Percent Complete{0:0.2f}%'.format(percent))
                print ('count: ', i)
            i = i+1
            
        ind = ind+1
            #print f1
            #print imList
    #saveTrainImages()    

    print ('done')
    f.close() # you can omit in most cases as the destructor will call it
    return j
if __name__=='__main__':
    Paths = ["E:/img/Animals/Rabbit/","E:/img/Animals/Racoon/","E:/img/Animals/Skunk/","E:/img/Animals/Weasel/","E:/img/Animals/Bobcat/","E:/img/Animals/Cat/","E:/img/Animals/Chicken/","E:/img/Animals/Coyote/","E:/img/Animals/Deer/","E:/img/Animals/Dog/","E:/img/Animals/Duck/","E:/img/Animals/Eagle/","E:/img/Animals/Hawk/","E:/img/Animals/MountainLion/","E:/img/Animals/Owl/","E:/img/Animals/Possum/","E:/img/Animals/Robin/","E:/img/Animals/Squirrel/","E:/img/Animals/Woodpecker/","E:/img/Animals/Humans/","E:/img/Animals/BlueJay/"  ]
    writeCategories(output, Paths)
