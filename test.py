import os
import time
import torch
import lpips
import argparse
import cv2 as cv
import numpy as np
import torchvision
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

import utils
import models

parser = argparse.ArgumentParser(description="Args of test.")
parser.add_argument("--rate", default=0.1, type=float)
parser.add_argument("--device", default="0")
opt_test = parser.parse_args()
opt_test.device = "cuda:" + opt_test.device

loss_fn_alex = lpips.LPIPS(net='alex')


def addsalt_pepper(img, SNR):
    b, c, h, w = img.size()
    mask = np.random.choice((0, 1, 2), size=(h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    img = img.squeeze()
    img[mask == 1] = 1.
    img[mask == 2] = 0.
    img = img.unsqueeze(0).unsqueeze(0)
    return img


def evaluate(config, net):
    if os.path.exists(config.model):
        if torch.cuda.is_available():
            trained_model = torch.load(config.model, map_location=config.device)
        else:
            trained_model = torch.load(config.model, map_location="cpu")

        net.load_state_dict(trained_model)
    else:
        raise FileNotFoundError("Missing trained models.")

    eva(config, net)


def eva(config, net, flag=False):
    f_names = ["Set5", ]  # "Set5", "Set14", "CBSD68", "Manga109"
    if flag:
        f_names = ["Set11", ]
    for folder in f_names:
        net = net.eval()
        noise = 'noise'
        sigmas = []
        st = 0
        if noise == 'noise':
            sigmas = [0, 0.001, 0.002, 0.005, 0.01]
        elif noise == 'SNR':
            sigmas = [0.99, 0.97, 0.95]
        for noise_sigma in sigmas:
            f_name = folder
            path = './dataset/test/' + f_name + "/"
            colors = ['g', 'r', 'b']
            mask = np.random.choice((0, 1, 2), size=(256, 256),
                                    p=[noise_sigma, (1 - noise_sigma) / 2., (1 - noise_sigma) / 2.])
            for clr in colors:
                with torch.no_grad():
                    for root, ds, fs in os.walk(path):
                        for f in fs:
                            print(clr + '    ' + f)
                            x = cv.imread(path + '/{}/'.format(clr) + f, flags=cv.IMREAD_GRAYSCALE)
                            x = torch.from_numpy(x / 255.).to(config.device).float()

                            h = x.size()[0]
                            h_lack = 0
                            w = x.size()[1]
                            w_lack = 0

                            if h % config.block_size != 0:
                                h_lack = config.block_size - h % config.block_size
                                temp_h = torch.zeros(h_lack, w).to(config.device)
                                h = h + h_lack
                                x = torch.cat((x, temp_h), 0)

                            if w % config.block_size != 0:
                                w_lack = config.block_size - w % config.block_size
                                temp_w = torch.zeros(h, w_lack).to(config.device)
                                w = w + w_lack
                                x = torch.cat((x, temp_w), 1)

                            x = torch.unsqueeze(x, 0)
                            x = torch.unsqueeze(x, 0)

                            idx_h = range(0, h, config.block_size)
                            idx_w = range(0, w, config.block_size)
                            num_patches = h * w // (config.block_size ** 2)

                            temp = torch.zeros(num_patches, 1, 1, config.block_size, config.block_size)
                            count = 0
                            for a in idx_h:
                                for b in idx_w:
                                    ori = x[:, :, a:a + config.block_size, b:b + config.block_size].to(config.device)
                                    if noise == 'noise':
                                        gauss = (noise_sigma ** 0.5) * torch.randn(1, 1, config.block_size,
                                                                                   config.block_size).to(config.device)
                                        ori = ori + gauss
                                    elif noise == 'SNR':
                                        ori = ori.squeeze()
                                        ori[mask == 1] = 1.
                                        ori[mask == 2] = 0.
                                        ori = ori.unsqueeze(0).unsqueeze(0)
                                    start = time.time()
                                    output, _ = net(ori)
                                    end = time.time()
                                    temp[count, :, :, :, :, ] = output
                                    count = count + 1
                                    st += end - start
                            y = torch.zeros(1, 1, h, w)

                            count = 0
                            for a in idx_h:
                                for b in idx_w:
                                    y[:, :, a:a + config.block_size, b:b + config.block_size] = temp[count, :, :, :, :]
                                    count = count + 1

                            recon = y[:, :, 0:h - h_lack, 0:w - w_lack]
                            recon = torch.squeeze(recon).to("cpu")

                            if not os.path.isdir("./gene_images/"):
                                os.mkdir("./gene_images/")
                            if not os.path.isdir("./gene_images/{}/".format(folder)):
                                os.mkdir("./gene_images/{}/".format(folder))
                            if not os.path.isdir("./gene_images/{}/rate{}_{}{}/".format(
                                    folder, int(config.ratio * 100), noise, int(noise_sigma * 1000))):
                                os.mkdir("./gene_images/{}/rate{}_{}{}/".format(
                                    folder, int(config.ratio * 100), noise, int(noise_sigma * 1000)))
                            img_path = "./gene_images/{}/rate{}_{}{}/{}/".format(
                                folder, int(config.ratio * 100), noise, int(noise_sigma * 1000), clr)
                            if not os.path.isdir(img_path):
                                os.mkdir(img_path)

                            tensor2image = torchvision.transforms.ToPILImage()
                            recon = tensor2image(recon)
                            recon.save(img_path + "{}.png".format(f.split('.')[0]))

            with torch.no_grad():
                c3_dir = "./gene_images/{}/rate{}_{}{}/".format(folder, int(config.ratio * 100), noise,
                                                                int(noise_sigma * 1000))
                if os.path.isfile(c3_dir + 'detail.txt'):
                    os.remove(c3_dir + 'detail.txt')
                if os.path.isfile(c3_dir + 'sum.txt'):
                    os.remove(c3_dir + 'sum.txt')
                p_sum, s_sum, l_sum = 0, 0, 0
                global img_num
                img_num = 0
                for root, ds, nfs in os.walk('./dataset/test/' + f_name + '/b/'):
                    img_num = len(nfs)
                    for f in nfs:
                        b = cv.imread(c3_dir + 'b/' + f, flags=cv.IMREAD_GRAYSCALE)
                        g = cv.imread(c3_dir + 'g/' + f, flags=cv.IMREAD_GRAYSCALE)
                        r = cv.imread(c3_dir + 'r/' + f, flags=cv.IMREAD_GRAYSCALE)
                        recon = cv.merge([b, g, r])
                        if not os.path.isdir(c3_dir + 'recon/'):
                            os.mkdir(c3_dir + 'recon/')
                        cv.imwrite(c3_dir + 'recon/' + f, recon)

                        if os.path.isfile(c3_dir + 'b/' + f):
                            os.remove(c3_dir + 'b/' + f)
                        if os.path.isfile(c3_dir + 'g/' + f):
                            os.remove(c3_dir + 'g/' + f)
                        if os.path.isfile(c3_dir + 'r/' + f):
                            os.remove(c3_dir + 'r/' + f)

                        img1 = cv.imread('./dataset/test/' + folder + '/' + f)
                        img2 = cv.imread(c3_dir + 'recon/' + f)
                        p = PSNR(img1, img2, data_range=255)
                        p_sum += p
                        s = SSIM(img1, img2, data_range=255, multichannel=True)
                        s_sum += s
                        l1 = lpips.im2tensor(lpips.load_image('./dataset/test/' + folder + '/' + f))
                        l2 = lpips.im2tensor(lpips.load_image(c3_dir + 'recon/' + f))
                        l = loss_fn_alex(l1, l2).squeeze().numpy()
                        l_sum += l

                        if not os.path.isfile(c3_dir + 'detail.txt'):
                            output_file = open(c3_dir + 'detail.txt', 'w')
                            output_file.write("=" * 120 + "\n")
                            output_file.close()
                        output_file = open(c3_dir + 'detail.txt', 'r+')
                        old = output_file.read()
                        output_file.seek(0)
                        output_file.write(old)
                        output_file.write("{:30s}, {:.4f}, {:.4f}, {:.4f}\n".
                                          format(f, p, s, l))
                        output_file.close()
                print("done.")
            if os.path.isdir(c3_dir + 'b/'):
                os.rmdir(c3_dir + 'b/')
            if os.path.isdir(c3_dir + 'g/'):
                os.rmdir(c3_dir + 'g/')
            if os.path.isdir(c3_dir + 'r/'):
                os.rmdir(c3_dir + 'r/')

            if not os.path.isfile(c3_dir + 'sum.txt'):
                output_file = open(c3_dir + 'sum.txt', 'w')
                output_file.write("=" * 120 + "\n")
                output_file.close()
            output_file = open(c3_dir + 'log.txt', 'r+')
            output_file.write("Res: PSNR ,{:.4f}, SSIM ,{:.4f}, LPIPS ,{:.4f},\n".
                              format(p_sum / img_num, s_sum / img_num, l_sum / img_num))
            output_file.close()
    if flag:
        return p_sum / img_num, s_sum / img_num


if __name__ == "__main__":
    # for rate in (0.01, 0.04, 0.1, 0.25):
    rate = opt_test.rate
    config = utils.GetConfig(ratio=rate, device=opt_test.device)
    net = models.Net(config).to(config.device).eval()
    evaluate(config, net)
