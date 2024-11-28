import torch
import torch.nn as nn

from Net import *
from Net.EEGConformer_net import Conformer


def build_net(args, shape):
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # for Mac
    elif torch.cuda.is_available():
        device = f"cuda:{args.gpu}"

    print(f"Using device: {device}")
    print("[Build Net]")

    if args.net == "EEGNet":
        net = EEGNet_net.EEGNet(args, shape)
        # load pretrained parameters
        if args.mode == "train":
            param = torch.load(f"./pretrained/{args.train_subject[0]-1}/checkpoint/500.tar", map_location=device)
            net.load_state_dict(param["net_state_dict"])

        # test only
        else:
            param = torch.load(f"./tl/{args.train_subject[0]-1}/checkpoint/50.tar", map_location=device)
            net.load_state_dict(param["net_state_dict"])
    elif args.net == "EEGConformer":
        net = Conformer()
        if args.mode == "train":
            param = torch.load(args.pretrained_path, map_location=device)
            net.load_state_dict(param["net_state_dict"])
    else:
        raise "args.net must be one of values ['EEGNet', 'EEGConformer']"

    # Set GPU
    if args.gpu != "cpu":
        check_gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
        assert check_gpu, "Check MPS or NVIDIA-GPU"
        if args.gpu == "multi":
            net = nn.DataParallel(net)
        net.to(device)

    # Set CPU
    else:
        device = torch.device("cpu")

    # Print
    print(f"device: {device}")
    print("")

    return net
