import torch
import torch.nn as nn

from Net import *
from Net.EEGConformer_net import Conformer


def build_net(args, shape):
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

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
    else:
        raise "args.net must be one of values ['EEGNet', 'EEGConformer']"

    # Set GPU
    if args.gpu != "cpu":
        assert torch.backends.mps.is_available(), "Check MPS"
        if args.gpu == "multi":
            net = nn.DataParallel(net)
        # else:
        #     torch.cuda.set_device(device)
        net.to(device)

    # Set CPU
    else:
        device = torch.device("cpu")

    # Print
    print(f"device: {device}")
    print("")

    return net
