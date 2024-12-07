import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from metrics import cal_log
from utils import print_update, createFolder, write_json, print_dict

torch.set_printoptions(linewidth=1000)


class Solver:
    def __init__(self, args, net, train_loader, val_loader, criterion, optimizer, scheduler, log_dict):
        self.args = args
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_dict = log_dict
        self.device = torch.device("cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = f"cuda:{args.gpu}"
        self.writer = SummaryWriter(log_dir=os.path.join(args.save_path, "tensorboard"))

    def train(self, epoch):
        log_tmp = {key: [] for key in self.log_dict.keys() if "train" in key}
        self.net.train()
        for i, data in enumerate(self.train_loader):
            # Load batch data
            inputs, task_labels, domain_labels = (
                data[0].to(self.device),
                data[1].to(self.device),
                data[2].to(self.device),
            )

            # Feed-forward
            self.optimizer.zero_grad()
            task_output, domain_output = self.net(inputs)
            task_loss = self.criterion(task_output, task_labels)
            domain_loss = self.criterion(domain_output, domain_labels)
            total_loss = task_loss + 0.1 * domain_loss

            # Backward
            total_loss.backward()
            self.optimizer.step()

            # Calculate log
            print(f"\ntask_loss: {task_loss.item():.4f}, domain_loss: {domain_loss.item():.4f}")
            cal_log(
                log_tmp,
                outputs=task_output,
                labels=task_labels,
                loss=total_loss,
            )

            # Print
            sentence = f"({(i + 1) * self.args.batch_size} / {len(self.train_loader.dataset.X)})"
            for key, value in log_tmp.items():
                sentence += f" {key}: {value[i]:0.3f}"
            print_update(sentence, i)
        print("")
        # Record log
        for key in log_tmp.keys():
            self.log_dict[key].append(np.mean(log_tmp[key]))
            self.writer.add_scalar(f"Train/{key}", self.log_dict[key][-1], epoch)

    def val(self, epoch):
        log_tmp = {key: [] for key in self.log_dict.keys() if "val" in key}
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                # Load batch data
                inputs, task_labels, domain_labels = (
                    data[0].to(self.device),
                    data[1].to(self.device),
                    data[2].to(self.device),
                )

                # Feed-forward
                task_output, domain_output = self.net(inputs)
                task_loss = self.criterion(task_output, task_labels)
                domain_loss = self.criterion(domain_output, domain_labels)
                total_loss = task_loss + domain_loss

                # Calculate log
                print(f"\ntask_loss: {task_loss.item():.4f}, domain_loss: {domain_loss.item():.4f}")
                cal_log(
                    log_tmp,
                    outputs=task_output,
                    labels=task_labels,
                    loss=total_loss,
                )

            # Record log
            for key in log_tmp.keys():
                self.log_dict[key].append(np.mean(log_tmp[key]))
                self.writer.add_scalar(f"Val/{key}", self.log_dict[key][-1], epoch)

    def experiment(self):
        print("[Start experiment]")
        total_epoch = self.args.epochs

        # freeze params of network except FC layer
        # if self.args.mode == "train":
        #     for name, param in self.net.named_parameters():
        #         if name != "linear.1.weight" and name != "linear.1.bias":
        #             param.requires_grad = False

        best_epoch = 0
        best_acc = -1e9
        for epoch in range(1, total_epoch + 1):
            print(f"Epoch {epoch}/{total_epoch}")
            # Train
            self.train(epoch)

            # Validation
            self.val(epoch)

            # Print
            print("=>", end=" ")
            for key, value in self.log_dict.items():
                print(f"{key}: {value[epoch - 1]:0.3f}", end=" ")
            print("")
            print(f"=> learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")

            # Update scheduler
            self.scheduler.step() if self.scheduler else None

            # Save checkpoint
            createFolder(os.path.join(self.args.save_path, "checkpoint"))
            if best_acc < self.log_dict["val_acc"][-1]:  # latest epoch
                best_acc = self.log_dict["val_acc"][-1]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": best_epoch,
                        "net_state_dict": self.net.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    },
                    os.path.join(self.args.save_path, f"checkpoint/best.tar"),
                )


        # Save args & log_dict
        self.args.seed = str(torch.manual_seed(self.args.seed))
        self.args.cuda_seed = self.args.seed
        self.args.acc = np.round(
            self.log_dict["val_acc"][-1], 3
        )  # NOTE: 학습 시작할 때 저장하거나, 다른 metric 있는 상황도 고려
        delattr(self.args, "topo") if hasattr(self.args, "topo") else None
        delattr(self.args, "phase") if hasattr(self.args, "phase") else None
        write_json(os.path.join(self.args.save_path, "args.json"), vars(self.args))

        print("====================================Finish====================================")
        print(self.net, "\n")
        print_dict(vars(self.args))
        print(f"Best acc: {best_acc}, epoch: {best_epoch}, checkpoint: {os.path.join(self.args.save_path, 'checkpoint/best.tar')}")

    def test(self):
        print("[Start test]")
        for epoch in range(1, 2):

            # Validation
            self.val(epoch)

            # Print
            print("=>", end=" ")
            print(f"test acc: {self.log_dict['val_acc']}")
        print("====================================Finish====================================")
