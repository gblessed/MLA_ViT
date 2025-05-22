import json
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import defaultdict
import torch
class MyObjectDetectionDataloader:

    def __init__(self, image_dir="Self-Driving-Car-3\export", annt_file = "Self-Driving-Car-3\export\_annotations.coco.json", train_transforms= None, img_w=512, img_h=512 ):

        self.train_transforms =  train_transforms
        self.image_dir = image_dir
        self.annt_file =  annt_file
        self.img_w   = img_w
        self.img_h = img_h
        with open(self.annt_file , "r") as f:
            self.annot_data = json.loads(f.read())

        self.class_idx_to_name = {cat['id']: cat['name'] for i, cat in enumerate(self.annot_data["categories"])}
        self.image_id_to_labels = defaultdict(list)
        for annotation in self.annot_data["annotations"]:
            self.image_id_to_labels[annotation["image_id"]].append(annotation)
         


    def __getitem__(self, idx):
        image_name = self.annot_data["images"][idx]['file_name']
        image_id = self.annot_data["images"][idx]['id']

        img_path = os.path.join(self.image_dir, image_name)
        image = Image.open(img_path).convert('RGB')

        labels = []
        boxes = []


        for item in self.image_id_to_labels[image_id]:
            x, y, w, h = item['bbox']  
            boxes.append([x/self.img_w, y/self.img_h, (x + w)/self.img_w,  (y + h)/self.img_h])
            labels.append([item['category_id']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if self.train_transforms:
            image = self.train_transforms(image)

        output_target  = {}
        output_target['boxes'] = boxes
        output_target['labels'] = labels

        return image, output_target


    def __len__(self):
        return len(self.annt_file)
        pass
    
def collate_fn(batch):
    print(len(batch))

    batch_image = torch.tensor([np.array(item[0]) for item in batch])
    batch_image = batch_image.permute(0,3, 2, 1)
    targets = [item[1] for item in batch]
    return batch_image, targets
