import torch, sys, os, argparse, utils, time, datetime
from torch import nn, optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.backends import cudnn
from tqdm import tqdm
from config import *
from datasets import CSDDataset, CityscapesDataset, CamvidDataset
from models.ccnet import CCNet
from models.pccnet import PCCNet
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_args_parser():
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument("--config", help="The config file", default=None, type=str)
    parser.add_argument("--seed", help="The random seed", default=42, type=int)
    parser.add_argument("--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--workers", default=2, type=int, help="number of data loading workers (default: 2)")
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--wd","--weight-decay", default=1e-4, type=float, help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run"),
    parser.add_argument("--output-dir", default="checkpoints", type=str, help="path to save checkpoints"),
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--restore-from", type=str, help="the restore chechpoint")
    return parser

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
    if len(losses) == 1:
        return losses['out']
    return losses['out'] + 0.5 * losses['aux']

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    pbar = tqdm(data_loader, file=sys.stdout, bar_format="{desc}{bar}[{elapsed}<{remaining},{rate_fmt}]")
    confmat = utils.ConfusionMatrix(num_classes)
    num_processed_samples = 0
    with torch.no_grad():
        for idx, (image, target) in enumerate(pbar):
            image, target = image.to(device), target.to(device)
            output = model(image)
            confmat.update(target.flatten(), output.argmax(1).flatten())
            num_processed_samples += image.shape[0]
    return confmat

def train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler=None):
    model.train()
    pbar =  tqdm(data_loader, file=sys.stdout, bar_format="{desc}[{elapsed}<{remaining},{rate_fmt}]")
    for idx, (image, target) in enumerate(pbar):
        image, target = image.to(device), target.to(device)
        loss = model(image, target) # Integrate loss function into model when model.training is True
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        print_str = "Epoch:{} Iter:{}/{}".format(epoch+1, idx, len(data_loader)) + " lr=%.4e" % optimizer.param_groups[0]['lr'] + " loss=%0.4f" % loss.item()
        pbar.set_description(print_str, refresh=False)

def main():
    args = get_args_parser().parse_args()
    # 设置随机种子，保证训练可复现
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    # 设置数据增强
    transforms = A.Compose([
        A.HorizontalFlip(0.5),
        A.RandomCrop(height=1024, width=1024),
        ToTensorV2(),
    ])
    # 获取数据集
    # dataset = CSDDataset(dataset_root='datasets/CSD', mode='train', transforms=transforms)
    # dataset_test = CSDDataset(dataset_root='datasets/CSD', mode='test', transforms=ToTensorV2())
    # dataset = CityscapesDataset(dataset_root='datasets/cityscapes', mode='train', transforms=transforms)
    # dataset_test = CityscapesDataset(dataset_root='datasets/cityscapes', mode='val', transforms=ToTensorV2())
    dataset = CamvidDataset(dataset_root='datasets/camvid', mode='train')
    dataset_test = CamvidDataset(dataset_root='datasets/camvid', mode='test')

    cudnn.benchmark = True # 不知道有什么用

    # 设置采样器/分布式训练采样器
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    # else:
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    # 设置数据读取器
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        drop_last=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers
    )
    # 构建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CCNet(num_classes=dataset.NUM_CLASSES).to(device)
    # NOTE: ccnet是预测NUM_CLASSES类别
    
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
    ]
    # set optimized parameters
    # params_to_optimize = [{"params": [p for p in model.backbone.parameters() if p.requires_grad]}]
    # for i in range(1, 5):
    #     params_to_optimize.append({"params": [p for p in getattr(model, f'classifier{i}').parameters() if p.requires_grad]})

    optimizer = optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    iters_per_epoch = len(data_loader)
    main_lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5000, gamma=0.8)

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_epochs.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # criterion = nn.CrossEntropyLoss(ignore_index=255)

    if args.restore_from:
        checkpoint = torch.load(args.restore_from)
        try:
            model.load_state_dict(checkpoint['model'])
        except RuntimeError as e:
            print(e)
            model.load_state_dict(checkpoint['model'], strict=False)
        args = checkpoint['args']
        args.start_epoch = checkpoint['epoch'] # 更新start_epoch
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("restore args ", args)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq)
        confmat = evaluate(model, data_loader_test, device=device, num_classes=dataset.NUM_CLASSES)
        print(confmat)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        # print(f"Saving checkpoint into {args.output_dir}")
        # os.makedirs(args.output_dir, exist_ok=True)
        # torch.save(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        # torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

if __name__ == '__main__':
    main()
