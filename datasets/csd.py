from torch.utils import data
import numpy as np
from PIL import Image
import os, cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CSDDataset(data.Dataset):
    NUM_CLASSES = 3
    IGNORE_INDEX = 255
    ID_TO_TRAINID = {0: IGNORE_INDEX, 38: 0, 75: 1, 113: 2}
    IMG_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMG_STD = np.array([0.229, 0.224, 0.225]) * 255
    def __init__(self, dataset_root, mode='train', transforms=None):
        self.dataset_root = dataset_root
        self.transforms = transforms
        self.mode = mode
        self.images = list()
        self.masks = list()
        
        position = os.listdir(dataset_root)
        for pos in position:
            imgs_list = os.listdir(os.path.join(dataset_root, pos, mode))
            images = [img for img in imgs_list if 'GT' not in img]
            masks = [img.split('.')[0]+'_GT.png' for img in images]
            self.images += [os.path.join(dataset_root, pos, mode, img) for img in images]
            self.masks += [os.path.join(dataset_root, pos, mode, mask) for mask in masks]

    def __len__(self):
        return len(self.images)
    
    def transform_id(self, target):
        mask = target.copy()
        for k, v in self.ID_TO_TRAINID.items():
            mask[target == k] = v
        return mask

    def __getitem__(self, index):
         # 读取图片文件和标签文件
        img = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        target = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        img = (img - self.IMG_MEAN) / self.IMG_STD
        target = self.transform_id(target=target)
        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=target)
            img = transformed['image']
            target = transformed['mask']
        return img.float(), target.long()
    
if __name__ == '__main__':
    transforms = A.Compose([
        A.HorizontalFlip(0.5),
        A.RandomCrop(height=1024, width=1024),
        ToTensorV2(),
    ])
    csd = CSDDataset(dataset_root='/opt/data/private/segmentation/datasets/CSD', mode='train', transforms=transforms)

    print(csd[0])
    pass
