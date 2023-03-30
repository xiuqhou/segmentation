from torch import nn
from torch.utils import data
import albumentations as A
import os, cv2
from albumentations.pytorch import ToTensorV2

class CamvidDataset(data.Dataset):
    IGNORE_INDEX = 255
    NUM_CLASSES = 11
    ID_TO_TRAINID = {-1: IGNORE_INDEX, 0: IGNORE_INDEX, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11}
    def __init__(self, dataset_root, mode='train', transforms=None):
        self.transforms = transforms
        if not transforms:
            self.transforms = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Normalize(),
                ToTensorV2(),            
            ])
        
        self.ids = os.listdir(os.path.join(dataset_root, mode))
        self.images = [os.path.join(dataset_root, mode, image_id) for image_id in self.ids]
        self.masks = [os.path.join(dataset_root, mode+'annot', mask_id) for mask_id in self.ids]
    
    def __len__(self):
        return len(self.images)

    def transform_id(self, target):
        mask = target.copy()
        for k, v in self.ID_TO_TRAINID.items():
            mask[target == k] = v
        return mask

    def __getitem__(self, index):
        img = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        target = self.transform_id(target=target)
        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=target)
            img = transformed['image']
            target = transformed['mask']
        return img.float(), target.long()