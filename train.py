import os
import sys
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import scipy.io as scio
import torch.optim as optim
import torchsummary.torchsummary
import torch.optim.lr_scheduler as LS

import models
import utils

from test import eva

parser = argparse.ArgumentParser(description="Args of this repo.")
parser.add_argument("--rate", default=0.1, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--batch", default=64, type=int)
parser.add_argument("--device", default="0")
parser.add_argument("--time", default=0, type=int)
opt_test = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    device = "cuda:" + opt_test.device
    config = utils.GetConfig(ratio=opt_test.rate, device=device)  # Set config by parser.
    config.check()  # mkdir.
    set_seed(22)  # seed.
    print("Data loading...")
    torch.cuda.empty_cache()
    dataset_train = utils.train_loader(config)
    net = models.Net(config).to(config.device)
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=10e-3)
    scheduler = LS.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    if os.path.exists(config.model):
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(config.model, map_location=config.device))
            info = torch.load(config.info, map_location=config.device)
        else:
            net.load_state_dict(torch.load(config.model, map_location="cpu"))
            info = torch.load(config.info, map_location="cpu")

        start_epoch = info["epoch"]
        best = info["res"]
        print("Loaded trained model of epoch {:2}, res: {:8.4f}.".format(start_epoch, best))
    else:
        start_epoch = 1
        best = 0
        print("No saved model, start epoch = 1.")

    # scheduler = LS.MultiStepLR(optimizer, milestones=[1, 19], gamma=0.1)

    over_all_time = time.time()
    for epoch in range(start_epoch, 100):
        print("Epoch: " + str(epoch))
        print("Lr: {}.".format(optimizer.param_groups[0]['lr']))

        epoch_loss = 0  # [epoch init loss, epoch deep loss]
        dic = {"rate": config.ratio, "epoch": epoch,
               "device": config.device, "lr": optimizer.param_groups[0]['lr']}
        for idx, xi in enumerate(tqdm(dataset_train, desc="Now training: ", postfix=dic)):
            xi = xi.to(config.device)

            optimizer.zero_grad()
            xo, _ = net(xi)
            batch_loss = torch.mean(torch.pow(xo - xi, 2)).to(config.device)

            if epoch != 1 and batch_loss > 2:
                print("\nWarning: your loss > 2 !")

            epoch_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                tqdm.write("\r[{:5}/{:5}], Loss: [{:8.6f}]"
                           .format(config.batch_size * (idx + 1),
                                   dataset_train.__len__() * config.batch_size,
                                   batch_loss.item()))

        avg_loss = epoch_loss / dataset_train.__len__()
        print("\n=> Epoch of {:2}, Epoch Loss: [{:8.6f}]"
              .format(epoch, avg_loss))

        # Make a log note.
        if epoch == 1:
            if not os.path.isfile(config.log):
                output_file = open(config.log, 'w')
                output_file.write("=" * 120 + "\n")
                output_file.close()
            output_file = open(config.log, 'r+')
            old = output_file.read()
            output_file.seek(0)
            output_file.write("\nAbove is {} test. Note：{}.\n"
                              .format("???", None) + "=" * 120 + "\n")
            output_file.write(old)
            output_file.close()

        # todo
        torch.save(net.state_dict(), config.second)
        p, s = eva(config, net, flag=True)
        print("{:5.3f}".format(p))
        if p > best:
            info = {"epoch": epoch, "res": p}
            torch.save(net.state_dict(), config.model)
            torch.save(optimizer.state_dict(), config.optimizer)
            torch.save(scheduler.state_dict(), config.scheduler)
            torch.save(info, config.info)
            print("*", "  Check point of epoch {:2} saved  ".format(epoch).center(120, "="), "*")
            best = p
            output_file = open(config.log, 'r+')
            old = output_file.read()
            output_file.seek(0)

            output_file.write("Epoch {:2.0f}, Loss of train {:8.6f}, Res {:2.4f}, {:2.4f}\n".format(
                epoch, avg_loss, best, s))
            output_file.write(old)
            output_file.close()

        scheduler.step()
        print("Over all time: {:.3f}s".format(time.time() - over_all_time))

    print("Train end.")


def gpu_info():
    memory = int(os.popen('nvidia-smi | grep %').read()
                 .split('C')[int(opt_test.device) + 1].split('|')[1].split('/')[0].split('MiB')[0].strip())
    return memory


if __name__ == "__main__":
    main()
