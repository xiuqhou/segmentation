from torch.utils import data
import os, cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
class CityscapesDataset(data.Dataset):
    IGNORE_INDEX = 255
    NUM_CLASSES = 19
    ID_TO_TRAINID = {-1: IGNORE_INDEX, 0: IGNORE_INDEX, 1: IGNORE_INDEX, 2: IGNORE_INDEX, 3: IGNORE_INDEX, 4: IGNORE_INDEX, 5: IGNORE_INDEX, 6: IGNORE_INDEX, 7: 0, 8: 1, 9: IGNORE_INDEX, 10: IGNORE_INDEX, 11: 2, 12: 3, 13: 4, 14: IGNORE_INDEX, 15: IGNORE_INDEX, 16: IGNORE_INDEX, 17: 5, 18: IGNORE_INDEX, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: IGNORE_INDEX, 30: IGNORE_INDEX, 31: 16, 32: 17, 33: 18}
    IMG_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMG_STD = np.array([0.229, 0.224, 0.225]) * 255
    def __init__(self, dataset_root, mode='train', transforms=None):
        self.dataset_root = dataset_root
        self.transforms = transforms
        self.mode = mode
        self.images = list()
        self.masks = list()

        mask_root = os.path.join(dataset_root, 'gtFine')
        mask_position = os.listdir(os.path.join(mask_root, mode))
        for pos in mask_position:
            mask_list = os.listdir(os.path.join(mask_root, mode, pos))
            masks = [mask for mask in mask_list if 'labelIds' in mask]
            images = [mask.replace("gtFine_labelIds","leftImg8bit") for mask in masks]
            self.masks += [os.path.join(mask_root, mode, pos, mask) for mask in masks]
            self.images += [os.path.join(dataset_root, 'leftImg8bit', mode, pos, img) for img in images]
        
    def __len__(self):
        return len(self.images)

    def transform_id(self, target):
        mask = target.copy()
        for k, v in self.ID_TO_TRAINID.items():
            mask[target == k] = v
        return mask
    
    def __getitem__(self, index):
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
        A.RandomCrop(height=1024, width=2048),
        ToTensorV2()
    ])
    cityscapes = CityscapesDataset(dataset_root='/opt/data/private/segmentation/datasets/cityscapes', mode='train', transforms=transforms)
    print(cityscapes[0])
    pass