import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DatDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.load_data()
        self.torch_form()

    def load_data(self):
        s = self.args.train_subject[0]  # Target subject
        if self.args.phase == "train":
            # Source domain: All subjects except the target subject
            X, y, domain_labels = [], [], []
            for subject_number in range(1, 10):
                domain_label = 1 if subject_number == s else 0  # Target: 1, Source: 0

                np_X = np.load(f"./data/S{subject_number:02}_train_X.npy")
                np_y = np.load(f"./data/S{subject_number:02}_train_y.npy")

                X.append(np_X)
                y.append(np_y)
                domain_labels.append(np.full(len(np_y), domain_label))  # Assign domain label to all samples

            # Concatenate all data
            self.X = np.concatenate(X)
            self.y = np.concatenate(y)
            self.domain_labels = np.concatenate(domain_labels)
        else:
            # Test data is from the target domain
            self.X = np.load(f"./data/S{s:02}_test_X.npy")
            self.y = np.load(f"./answer/S{s:02}_y_test.npy")
            self.domain_labels = np.full(len(self.y), 1)  # All test data belongs to the target domain

        # Ensure the data has a channel dimension
        if len(self.X.shape) <= 3:
            self.X = np.expand_dims(self.X, axis=1)

    def torch_form(self):
        self.X = torch.FloatTensor(self.X)  # EEG data
        self.y = torch.LongTensor(self.y)  # Task labels
        self.domain_labels = torch.LongTensor(self.domain_labels)  # Domain labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            sample (tuple): (EEG data, task label, domain label)
        """
        return self.X[idx], self.y[idx], self.domain_labels[idx]


class PretrainDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.load_data()
        # self.preprocess()
        self.torch_form()

    def load_np_data(self, s, session="1"):
        assert session in ["1", "2"]
        data_type = "train" if session == "1" else "test"
        X_path = f"./data/S{s:02}_{data_type}_X.npy"
        y_path = f"./data/S{s:02}_train_y.npy" if data_type == "train" else f"./answer/S{s:02}_y_test.npy"
        return np.load(X_path), np.load(y_path)

    def load_data(self):
        s = self.args.train_subject[0]
        if self.args.phase == "train":
            X, y = [], []
            for subject_number in range(1, 10):
                if not self.args.load_all or s != subject_number:
                    np_X, np_y = self.load_np_data(subject_number, "1")
                    X.append(np_X)
                    y.append(np_y)
                    if self.args.load_all:
                        np_X, np_y = self.load_np_data(subject_number, "2")
                        X.append(np_X)
                        y.append(np_y)

            self.X = np.concatenate(X)
            self.y = np.concatenate(y)
        else:
            self.X = np.load(f"./data/S{s:02}_test_X.npy")
            self.y = np.load(f"./answer/S{s:02}_y_test.npy")
        if len(self.X.shape) <= 3:
            self.X = np.expand_dims(self.X, axis=1)

    def preprocess(self):
        if self.args.phase == "train":
            aug_data, aug_label = self.interaug(self.X, self.y)
            self.X, self.y = aug_data, aug_label

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]
            if tmp_data.shape[0] == 0:
                print(f"cls_idx: {cls_idx}, tmp_data shape[0] == 0. skip interaug")
                continue

            tmp_aug_data = np.zeros((int(self.args.batch_size / 4), 1, 22, 1000))
            for ri in range(int(self.args.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125 : (rj + 1) * 125] = tmp_data[
                        rand_idx[rj], :, :, rj * 125 : (rj + 1) * 125
                    ]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[: int(self.args.batch_size / 4)])

        if len(aug_data) == 0:
            raise ValueError("No Augmented data generated")
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        return aug_data, aug_label

    def torch_form(self):
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = [self.X[idx], self.y[idx]]
        return sample


class CustomDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.load_data()
        self.torch_form()

    def load_data(self):
        s = self.args.train_subject[0]
        if self.args.phase == "train":
            self.X = np.load(f"./data/S{s:02}_train_X.npy")
            self.y = np.load(f"./data/S{s:02}_train_y.npy")
        else:
            self.X = np.load(f"./data/S{s:02}_test_X.npy")
            self.y = np.load(f"./answer/S{s:02}_y_test.npy")
        if len(self.X.shape) <= 3:
            self.X = np.expand_dims(self.X, axis=1)

    def torch_form(self):
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = [self.X[idx], self.y[idx]]
        return sample


def data_loader(args):
    print("[Load data]")
    # Load train data
    args.phase = "train"
    if args.net == "DatEEGNet":
        dataset_type = DatDataset
    else:
        dataset_type = PretrainDataset if args.mode == "pretrain" else CustomDataset
    trainset = dataset_type(args)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Load val data
    args.phase = "val"
    valset = dataset_type(args)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Print
    print(f"train_set size: {train_loader.dataset.X.shape}")
    print(f"val_set size: {val_loader.dataset.X.shape}")
    print("")
    return train_loader, val_loader
