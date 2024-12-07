import torch
import torch.nn as nn

from Net import *
from Net.EEGNet_net import EEGNet
from Net.DatEEGNet_net import EEGNetWithDAT
from Net.EEGConformer_net import Conformer
from Net.ATCNet_net import ATCNet


def build_net(args, shape):
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # for Mac
    elif torch.cuda.is_available():
        device = f"cuda:{args.gpu}"

    print(f"Using device: {device}")
    print("[Build Net]")

    if args.net == "EEGNet":
        net = EEGNet(args, shape)
        # load pretrained parameters
        if args.mode == "train":
            param = torch.load(f"./pretrained/{args.train_subject[0]-1}/checkpoint/500.tar", map_location=device)
            net.load_state_dict(param["net_state_dict"])

        # test only
        else:
            param = torch.load(f"./tl/{args.train_subject[0]-1}/checkpoint/50.tar", map_location=device)
            net.load_state_dict(param["net_state_dict"])
    elif args.net == "DatEEGNet":
        net = EEGNetWithDAT(args, shape)
        if args.mode == "test":
            param = torch.load(f"{args.pretrained_path}", map_location=device)
            net.load_state_dict(param["net_state_dict"])

    elif args.net == "EEGConformer":
        net = Conformer()
    elif args.net == "ATCNet":
        net = ATCNet()
    elif args.net == "FBCNet":
        config = {
            "nChan": 22,
            "nTime": 1000,
            "dropoutP": 0.5,
            "nBands": 1,
            "m": 32,
            "temporalLayer": "LogVarLayer",
            "nClass": 4,
            "doWeightNorm": True,
        }
        net = FBCNet(nChan=config["nChan"], nBands=config["nBands"])
    else:
        raise "args.net must be one of values ['EEGNet', 'EEGConformer', 'ATCNet', 'FBCNet']"

    if args.net not in ["EEGNet", "DatEEGNet"] and args.mode == "train":
        param = torch.load(args.pretrained_path, map_location=device)
        net.load_state_dict(param["net_state_dict"])
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
