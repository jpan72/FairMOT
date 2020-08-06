import argparse
import json
import time

from lib.models import *
# from lib.utils.datasets import GhostDataset, collate_fn
from lib.datasets.dataset.ghost_dataset import GhostDataset
from lib.utils.utils import *
from torchvision.transforms import transforms as T
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import os.path as osp

from lib.models.GPN_model import GPN
import torch
import torch.nn as nn
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def normalize_bbox(bbox, format):
    num_dim = len(bbox.shape)
    if num_dim == 2:
        bbox[:, 0] /= 1088
        bbox[:, 1] /= 608
        if format == "tlwh" or "tlbr":
            bbox[:, 2] /= 1088
        bbox[:, 3] /= 608
    elif num_dim == 3:
        bbox[:, :, 0] /= 1088
        bbox[:, ;, 1] /= 608
        if format == "tlwh" or "tlbr":
            bbox[:, :, 2] /= 1088
        bbox[:, :, 3] /= 608
    else:
        ValueError("Wrong input to normalize bbox")

    return bbox

def tlbrs_to_tlwhs(tlbrs):
    ret = tlbrs
    ret[...,2:] -= ret[...,:2]
    return ret

def tlwhs_to_xyahs(tlwhs):
    """Convert bounding boxes to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = tlwhs
    ret[...,:2] += ret[...,2:] / 2
    ret[...,2] /= ret[...,3]
    return ret

def train(
        cfg,
        data_cfg,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        freeze_backbone=False,
        opt=None,
):

    # Training configs
    weights = 'weights'
    if not osp.exists(weights):
        os.mkdir(weights)
    latest = osp.join(weights, 'latest.pt')
    torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Transform
    if opt.network == 'alexnet':
        input_size = 256
    else:
        input_size = 224
    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((input_size, input_size)),
        T.RandomCrop(200),
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataloader
    dataset_root = '../preprocess-ghost-bbox-th0.6-map-more-filter/MOT17/MOT17/train'
    dataset = GhostDataset(dataset_root, transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    print("Size of training data is {}".format(len(dataloader)))

    dataset_root_test = '../preprocess-ghost-bbox-th0.6-map-more-filter/2DMOT2015/train'
    dataset_test = GhostDataset(dataset_root_test, transforms)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=8)
    print("Size of test data is {}".format(len(dataloader_test)))

    # Initialize model
    gpn = GPN(network=opt.network).cuda()
    if opt.resume:
        gpn.load_state_dict(torch.load(opt.load_path))
    else:
        import torch.nn.init as weight_init
        for name, param in gpn.named_parameters():
            if 'weight' in name:
                weight_init.normal(param);

    # Optimizer and loss
    start_epoch = 0
    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, gpn.parameters()), lr=opt.lr, momentum=.9,
                                    weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, gpn.parameters()), lr=opt.lr)
    smooth_l1_loss = nn.SmoothL1Loss().cuda()
    smooth_l1_loss_test = nn.SmoothL1Loss(reduction='sum').cuda()

    # # If multi GPUs are needed
    # model = torch.nn.DataParallel(model)
    # # Scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #         milestones=[int(0.5*opt.epochs), int(0.75*opt.epochs)], gamma=0.1)
    # model_info(model)
    # model_info(gpn)

    # Tensorboard
    exp_name = f'{datetime.now():%Y-%m-%d-%H:%M%z}'
    writer = SummaryWriter(osp.join('../exp-ghost-bbox', exp_name))
    t0 = time.time()

    gpn_option = "absolute"  # "absolute" or "relevant"

    # Run training
    for epoch in range(epochs):
        print('Epoch {}'.format(epoch))

        # training
        gpn.train()
        epoch += start_epoch
        optimizer.zero_grad()

        for i, (track_imgs, det_imgs, tracks_tlbr, dets_tlbr, histories_tlwh, target_delta_bbox_tlwh) in enumerate(
                dataloader):

            n_iter = epoch * len(dataloader) + i

            # Normalize
            tracks_tlbr = normalize_bbox(tracks_tlbr, "tlbr")
            dets_tlbr = normalize_bbox(dets_tlbr, "tlbr")
            histories_tlwh = normalize_bbox(histories_tlwh, "tlwh")
            target_delta_bbox_tlwh = normalize_bbox(target_delta_bbox_tlwh, "tlwh")

            # Convert for tracks_xyah, dets_xyah
            tracks_tlwh = tlbrs_to_tlwhs(tracks_tlbr)
            tracks_xyah = tlwhs_to_xyahs(tracks_tlwh)
            dets_tlwh = tlbrs_to_tlwhs(dets_tlbr)
            dets_xyah = tlwhs_to_xyahs(dets_tlwh)
            histories_xyah = tlwhs_to_xyahs(histories_tlwh)

            # Restore FNs_tlwh and convert to FNs_xyah
            # in order to get target_bbox_xyah and target_delta_bbox_xyah
            FNs_tlwh = tracks_tlwh + target_delta_bbox_tlwh
            FNs_xyah = tlwhs_to_xyahs(FNs_tlwh)
            target_bbox_xyah = FNs_xyah
            target_delta_bbox_xyah = FNs_xyah - tracks_xyah
            
            # Move inputs and targets to GPU
            track_imgs = track_imgs.cuda().float()
            det_imgs = det_imgs.cuda().float()
            tracks_xyah = tracks_xyah.cuda().float()
            dets_xyah = dets_xyah.cuda().float()
            histories_xyah = histories_xyah.cuda().float()

            if gpn_option == "absolute":
                gpn_target = target_bbox_xyah
            elif gpn_option == "relevant":
                gpn_target = target_delta_bbox_xyah
            else:
                ValueError("gpu_option must be `absolute` or `relevant`")
            gpn_target = gpn_target.cuda().float()

            # Run GPN and computer loss
            gpn_output = gpn(track_imgs, det_imgs, tracks_xyah, dets_xyah, histories_xyah)
            loss = smooth_l1_loss(gpn_output, gpn_target)

            # Compute gradient
            loss.backward()
            writer.add_scalar('train/loss', loss.cpu().detach().numpy(), n_iter)

            # Accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            if i % 100 == 0:
                print()
                print('========= Train ==========')
                print("Output: {}".format(gpn_output))
                print("Target: {}".format(gpn_target))
                print("Loss: {}".format(loss))

        # Run evaluation
        gpn.eval()
        loss_test_sum = 0
        for i, (track_imgs, det_imgs, tracks_tlbr, dets_tlbr, histories_tlwh, target_delta_bbox_tlwh) in enumerate(
                dataloader_test):

            n_iter = epoch * len(dataloader_test) + i

            # Normalize
            tracks_tlbr = normalize_bbox(tracks_tlbr, "tlbr")
            dets_tlbr = normalize_bbox(dets_tlbr, "tlbr")
            histories_tlwh = normalize_bbox(histories_tlwh, "tlwh")
            target_delta_bbox_tlwh = normalize_bbox(target_delta_bbox_tlwh, "tlwh")

            # Convert for tracks_xyah, dets_xyah
            tracks_tlwh = tlbrs_to_tlwhs(tracks_tlbr)
            tracks_xyah = tlwhs_to_xyahs(tracks_tlwh)
            dets_tlwh = tlbrs_to_tlwhs(dets_tlbr)
            dets_xyah = tlwhs_to_xyahs(dets_tlwh)
            histories_xyah = tlwhs_to_xyahs(histories_tlwh)

            # Restore FNs_tlwh and convert to FNs_xyah
            # in order to get target_bbox_xyah and target_delta_bbox_xyah
            FNs_tlwh = tracks_tlwh + target_delta_bbox_tlwh
            FNs_xyah = tlwhs_to_xyahs(FNs_tlwh)
            target_bbox_xyah = FNs_xyah
            target_delta_bbox_xyah = FNs_xyah - tracks_xyah

            # Move inputs and targets to GPU
            track_imgs = track_imgs.cuda().float()
            det_imgs = det_imgs.cuda().float()
            tracks_xyah = tracks_xyah.cuda().float()
            dets_xyah = dets_xyah.cuda().float()
            histories_xyah = histories_xyah.cuda().float()

            if gpn_option == "absolute":
                gpn_target = target_bbox_xyah
            elif gpn_option == "relevant":
                gpn_target = target_delta_bbox_xyah
            else:
                ValueError("gpu_option must be `absolute` or `relevant`")
            gpn_target = gpn_target.cuda().float()

            # Run GPN and computer loss
            gpn_output = gpn(track_imgs, det_imgs, tracks_xyah, dets_xyah, histories_xyah)
            loss = smooth_l1_loss(gpn_output, gpn_target)
            loss_test_sum += loss.cpu().detach().numpy()

            if i % 100 == 0:
                print()
                print('========= Test ==========')
                print("Output: {}".format(gpn_output))
                print("Target: {}".format(gpn_target))
                print("Loss: {}".format(loss))

        loss_test_mean = loss_test_sum / len(dataloader_test)
        writer.add_scalar('test/loss', loss_test_mean, n_iter)

    writer.close()
    torch.save(gpn.state_dict(), opt.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=2, help='number of batches before optimizer step')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe.json', help='coco.data file path')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--print-interval', type=int, default=40, help='print interval')
    parser.add_argument('--test-interval', type=int, default=9, help='test interval')
    parser.add_argument('--lr', type=float, default=1e-2, help='init lr')
    parser.add_argument('--unfreeze-bn', action='store_true', help='unfreeze bn')
    parser.add_argument('--network', type=str, default='alexnet', help='alexnet or resnet')
    parser.add_argument('--optim', type=str, default='sgd', help='optimizer')
    parser.add_argument('--save-path', type=str, default='model.pth', help='model path')
    parser.add_argument('--load-path', type=str, default='model.pth', help='path to load model')

    opt = parser.parse_args()

    # init_seeds()

    train(
        opt.cfg,
        opt.data_cfg,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        opt=opt,
    )