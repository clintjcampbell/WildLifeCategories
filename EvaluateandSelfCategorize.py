# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:40:13 2016

@author: yg155d
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
import cntk
import _cntk_py
from cntk.ops.functions import load_model
import matplotlib.pyplot as plt
#import cntk as ct
directory = 'E:/img/Animals/'
output = directory+'Animals.txt'
imDimWd = 100
imDimTl = 100
def create_reader(map_file, mean_file, is_training):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("File '%s' or '%s' does not exist. Please run install_cifar10.py from DataSets/CIFAR-10 to fetch them" %
                           (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
        transforms += [
            cntk.io.ImageDeserializer.crop(crop_type='randomside', side_ratio=0.8, jitter_type='uniratio') # train uses jitter
        ]
    transforms += [
        cntk.io.ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        cntk.io.ImageDeserializer.mean(mean_file)
    ]
    # deserializer
    return cntk.io.MinibatchSource(cntk.io.ImageDeserializer(map_file, cntk.io.StreamDefs(
        features = cntk.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = cntk.io.StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=is_training)
            
def resizeImage(imgPath,savePath):
    img = Image.open(imgPath)
    img = img.resize((imDimWd,imDimTl),Image.ANTIALIAS)
    img.save(savePath)
def flattenImages(imgPath):
    img = Image.open(imgPath)
    img = img.resize((imDimWd,imDimTl),Image.ANTIALIAS)
    rgb_image = np.asarray(img, dtype=np.float32) - 128
    bgr_image = rgb_image[..., [2, 1, 0]]
    pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

    return pic
    #img.save(imgPath2)
#    f1 = im2.getdata()
#    imList = list(img.getdata())
#    #mode = img.mode    
#    #sizeIm = img.size
#    rlist = []
#    glist = []
#    blist = []
#    count = len(imList)
#    for i in range(0,count):
#    #for List in imList:
#    #    for i in List:
#        r,g,b = imList[i]
#        rlist.append(r)
#        glist.append(g)
#        blist.append(b)
#        
#    imgVec = rlist + glist + blist
#    return imgVec
    
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

#need to specify directories to search
#need to parse out input and output images
def Recategorize():
    print ('startingRecategorization')
    #f = open(output,'a+')
    #reader_test  = create_reader(os.path.join(data_path, 'Animals.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False)
    Paths = ["E:/img/Animals2/Rabbit/","E:/img/Animals2/Racoon/","E:/img/Animals2/Skunk/","E:/img/Animals2/Weasel/","E:/img/Animals2/Bobcat/","E:/img/Animals2/Cat/","E:/img/Animals2/Chicken/","E:/img/Animals2/Coyote/","E:/img/Animals2/Deer/","E:/img/Animals2/Dog/","E:/img/Animals2/Duck/","E:/img/Animals2/Eagle/","E:/img/Animals2/Hawk/","E:/img/Animals2/MountainLion/","E:/img/Animals2/Owl/","E:/img/Animals2/Possum/","E:/img/Animals2/Robin/","E:/img/Animals2/Squirrel/","E:/img/Animals2/Woodpecker/","E:/img/Animals2/Humans/","E:/img/Animals2/BlueJay/","E:/img/Animals2/Backgrounds/","E:/img/Animals2/Bear/" ]
    #Paths = ["E:/img/Animals2/Backgrounds/"]
    CategoriesPass=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    CategoriesFail=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    MisCategoriesFail=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ind = 0
    j = 0
    z = load_model("E:/img/Animals2/Models/ConvNet_CIFAR10_DataAugModelEvolveModel.dnn")
    for mypath in Paths[0:23]:   
        print ('path:' + mypath)
        onlyfiles = [fl for fl in listdir(mypath) if fl.endswith('.jpeg')]
        length = len(onlyfiles)
        i = 0
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
            #resizeImage(pth2,directory + "Animals/" + fl2)
            #img = Image.open(directory + "Animals/" + fl2)
            #pic = np.array(img, dtype=np.float32) 
            #img.show()
            #pic = np.ascontiguousarray(np.transpose(pic, (2, 0, 1)))
            #print (pic)
            pic = flattenImages(mypath + fl2)
            #evalu = z.eval(pic)
            #print (evalu)
            predictions = np.squeeze(z.eval(pic))
            top_class = np.argmax(predictions)
            #print (mypath,top_class,predictions)

            if top_class == ind:
                #img = pic.reshape([100,100,3])
                #imgplot = plt.imshow(img)
                #print(Paths[top_class])
                #print ('count: ', i)
                #plt.ion()
                #plt.pause(0.2)
                #plt.show()
                #plt.close()
            #else: 
                #Percent = 100*((predictions[ind])/(sum(predictions[0:21])))
                #print("Success!!!!!!!!!!!!!!!!!!!!!!!!!!", Percent)
                #print (mypath,top_class,predictions)
                #print(pth2)
                CategoriesPass[ind] = CategoriesPass[ind]+1
            else:
                #Percent = 100*((predictions[ind])/(sum(predictions[0:23])))
                #PercentT =100*((predictions[top_class])/(sum(predictions[0:15])))
                #print("Fail!", Percent, PercentT, Paths[top_class])
                CategoriesFail[ind] = CategoriesFail[ind]+1
                MisCategoriesFail[top_class] = MisCategoriesFail[top_class]+1
                if (predictions[ind] <=0):
                    print("SuperFail!", mypath, predictions, predictions[ind])
                    #img = Image.open(mypath + fl2)
                    #move really bad to processing
                    try:
                        os.rename(pth2, Paths[top_class]+ fl2)
                    except:
                        continue
                    #img.save("E:/img/Animals/Rejected/" + Paths[top_class][15:18] + fl2 )
                    #imgplot = plt.imshow(img)
                    #imgplot.fname = fl2
                    print(Paths[top_class])
                    print ('count: ', i)
                    #plt.ion()
                    #plt.pause(.1)
                    #plt.show()
                    #plt.close()

            i = i+1
            j = j+1
        ind = ind+1
 
    print (CategoriesPass)
    print (CategoriesFail)
    print(MisCategoriesFail)
    print ('done')
    #f.close() # you can omit in most cases as the destructor will call it


