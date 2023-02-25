import argparse, torch, time, sys, os, datetime
from utils import ConfusionMatrix, get_palette
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import CSDDataset, CityscapesDataset
from torch.backends import cudnn
from models.ccnet import CCNet
from tqdm import tqdm
from PIL import Image


def get_args_parser():
    parser = argparse.ArgumentParser(description='Model test')
    parser.add_argument("--config", help="The config file", default=None, type=str)
    parser.add_argument("--seed", help="The random seed", default=42, type=int)
    parser.add_argument("--batch-size", default=1, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--restore-from", default=None, type=str, help="where restore model parameters from")
    parser.add_argument("--save-dir", default="results", type=str, help="path to save outputs")
    return parser

def main():
    args = get_args_parser().parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    # dataset_test = CSDDataset(dataset_root='datasets/CSD', mode='test', transforms=ToTensorV2())
    dataset_test = CityscapesDataset(dataset_root='datasets/cityscapes', mode='val', transforms=ToTensorV2())
    cudnn.benchmark = True
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CCNet(num_classes=dataset_test.NUM_CLASSES+1).to(device)
    # TODO: load_from
    if args.restore_from:
        model.load_state_dict(torch.load(args.restore_from)['model'], strict=False)
    start_time = time.time()
    # TODO: 下面是evaluate方法的内容，考虑合并一下
    model.eval()
    pbar = tqdm(data_loader_test, file=sys.stdout, bar_format="{desc}{bar}[{elapsed}<{remaining},{rate_fmt}]")
    confmat = ConfusionMatrix(dataset_test.NUM_CLASSES+1)
    num_processed_samples = 0
    output_list = list()
    print("Starting test ...")
    with torch.no_grad():
        for idx, (image, target) in enumerate(pbar):
            image, target = image.to(device), target.to(device)
            output = model(image)
            # output_list.append(output)
            confmat.update(target.flatten(), output.argmax(1).flatten())
            num_processed_samples += image.shape[0]
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Testing time {total_time_str}")
    print(confmat)
    # 设置颜色，保存mask
    for idx, output in tqdm(enumerate(output_list)):
        output = output.argmax(1).cpu().numpy().squeeze().astype(np.uint8)
        output_im = Image.fromarray(output)
        output_im.putpalette(get_palette(dataset_test.num_classes+1))
        output_im = output_im.convert('L')
        os.makedirs(args.save_dir, exist_ok=True)
        output_im.save(os.path.join(args.save_dir, 
                                    os.path.split(dataset_test.images[idx])[-1]))

if __name__ == '__main__':
    main()