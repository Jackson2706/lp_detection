from torch.utils.data import Dataset, DataLoader
import os.path as osp
import glob
import xml.etree.ElementTree as ET
import cv2 
import torch

class licensePlateDataset(Dataset):
    def __init__(self, root = "../data", phase="train", transform = None):
        img_path = osp.join(root+"/"+phase+"/*.jpg")
        xml_path = osp.join(root+"/"+phase+"/*.xml")

        self.images_list = []
        self.gt_list = []

        for path in glob.glob(img_path):
            self.images_list.append(path)
        
        for path in glob.glob(xml_path):
            tree=ET.parse(path)
            roots=tree.getroot()
            labels = roots.find('object').find('name').text          
            if labels == "License-plate":
                labels = 1
            else: 
                labels = 0  
            xmins = int(roots.find('object').find('bndbox').find('xmin').text)
            ymins = int(roots.find('object').find('bndbox').find('ymin').text)
            xmaxs = int(roots.find('object').find('bndbox').find('xmax').text)
            ymaxs = int(roots.find('object').find('bndbox').find('ymax').text)
            gt = [xmins, ymins, xmaxs, ymaxs, labels]

            self.gt_list.append(gt)

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index):
        gt = self.gt_list[index]
        img = cv2.imread(self.images_list[index])
        img = cv2.resize(img, (300,300))
        img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)

        return img, gt
    
def my_collate_fn(batch):
        targets = []
        imgs = []

        for sample in batch:
            imgs.append(sample[0])
            targets.append(torch.FloatTensor(sample[1]))

        imgs = torch.stack(imgs, dim=0)

        return imgs, targets

if __name__ == "__main__":
    data = licensePlateDataset(root="data")
    print(data.__getitem__(1))

    train_dataset = licensePlateDataset(root="data", phase="train")
    val_dataset = licensePlateDataset(root="data", phase="valid")
    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    batch_iter = iter(dataloader_dict["val"])
    images, targets = next(batch_iter) # get 1 sample
    print(images.size()) 
    print(len(targets))
    print(targets[0].size())

        