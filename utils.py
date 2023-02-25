from collections import defaultdict, deque
import torch

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
    
    def update(self, a, b):
        """
        Calculate the confusion matrix based on ground truth and prediction
        Args:
            a: Flattened ground truth mask with shape (h, w) -> h*w
            b: Predicted mask with the same shape
        """
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n) # 排除ignore_index?
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)
    
    def reset(self):
        self.mat.zero_()
    
    def compute(self):
        # NOTE: Add small epsilon 1e-10 for zore divide error
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / (h.sum()  + 1e-10)
        acc = torch.diag(h) / (h.sum(1) + 1e-10)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-10)
        return acc_global, acc, iu

    def __str__(self) -> str:
        acc_global, acc, iu = self.compute()
        return ("""global correct: {:.2f}\naverage row correct: {}\nIoU: {}\nmean IoU: {:.2f}""".format(
            acc_global.item() * 100,
            [f"{i:.1f}" for i in (acc * 100).tolist()],
            [f"{i:.1f}" for i in (iu * 100).tolist()],
            iu.mean().item() * 100
        ))