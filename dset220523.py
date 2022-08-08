import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from collections import Set
import os
import pandas as pd
from torchvision.io import read_image
from csv import reader







class CustomImageDataset220523(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        label_set = set()
        with open(annotations_file, 'r') as read_obj:
            csv_reader = reader(read_obj)
            for path, Jp, En in csv_reader:
                label_set.add(En)
        label_list = list(label_set)

        self.label_dict = {}

        for i, label in enumerate(label_list):
            self.label_dict[label] = i
        print(label_list)
        self.classes = label_list
        print(self.label_dict)


        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = read_image(img_path)
        label = self.label_dict[self.img_labels.iloc[idx, 2]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == '__main__':
    CD = CustomImageDataset220523(annotations_file='220523FixedCSV.csv')

    print(CD[3])