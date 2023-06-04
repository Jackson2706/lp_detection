from torch.utils.data import Dataset
import os.path as osp
from PIL import Image
import glob
import torch
import numpy as np
import cv2
import xml.etree.ElementTree as ET
class dataset(Dataset):
    def __init__(self,root=".",phase="train",Tranform=None):
        img_path=osp.join(root+"/"+phase+"/*.jpg")
        xml_path=osp.join(root+"/"+phase+"/*.xml")
        self.image=[]
        self.xmin =[]
        self.ymin =[]
        self.xmax =[]
        self.ymax =[]
        self.label=[] 
        self.name = []
        for path in glob.glob(img_path):
            self.image.append(path)
        
        for path in glob.glob(xml_path):
            tree=ET.parse(path)
            roots=tree.getroot()

            filename=roots.find('filename').text
            labels = roots.find('object').find('name').text
            self.label.append(labels)
            
            self.name.append(filename)
            xmins = int(roots.find('object').find('bndbox').find('xmin').text)
            self.xmin.append(xmins)
            ymins = int(roots.find('object').find('bndbox').find('ymin').text)
            self.ymin.append(ymins)
            xmaxs = int(roots.find('object').find('bndbox').find('xmax').text)
            self.xmax.append(xmaxs)
            ymaxs = int(roots.find('object').find('bndbox').find('ymax').text)
            self.ymax.append(ymaxs)
            
        #print(len(self.image))
        #print(len(self.label))
        #print(len(self.xmax))

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self,index):
        imagepath= self.image[index]
        image = Image.open(self.image[index])
        image.load()
        image = np.asarray(image, dtype="int32")
        image = torch.from_numpy(image)
        label = self.label[index]
        name = self.name[index]
        xmin = self.xmin[index]
        ymin = self.ymin[index]
        xmax = self.xmax[index]
        ymax = self.ymax[index]
        return  image, label, name, xmin, xmax,  ymin,ymax 

        

        


if __name__=="__main__":
    data=dataset(root="./License_Plate.v4i.voc")
    image, label, name, xmin, xmax, ymin, ymax=data.__getitem__(1)
    #print(imagepath)
    print(image)
    print(label)
    print("filename",name)
    print("xmin",xmin)
    print("ymin",ymin)
    print("xmax",xmax)
    print("ymax:",ymax)



       

        




