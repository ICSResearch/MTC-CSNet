# MTC-CSNet
This repository is the `PyTtorch` code for our paper `"MTC-CSNet: Marrying Transformer and Convolution for Image Compressed Sensing"`.  
## 1. Introduction ##
**1) Datasets**  
aasssssssssssssssssssssssssssss
Training set: [`BSDS500`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) and [`VOC2012`](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/), validation set: [`Set11`](https://github.com/KuldeepKulkarni/ReconNet), testing sets: [`Set5`](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html), [`Set14`](https://huggingface.co/datasets/eugenesiow/Set14), [`CBSD68`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) and [`Manga109`](http://www.manga109.org/en/).  

**2）Project structure**
```
(MTC-CSNet)
|-dataset
|    |-train  
|        |-BSDS500 (.jpg)  
|        |-VOC2012 (.jpg)  
|    |-val  
|        |-Set11  
|            |-r (.png)  
|            |-g (.png)  
|            |-b (.png) 
|            |-(.png)  
|    |-test  
|        |-Set5  
|            |-r (.png)  
|            |-g (.png)  
|            |-b (.png) 
|            |-(.png)  
|        |-Set14 (.png)  
|            |-... (Same as Set5)  
|        |-CBSD68 (.png)  
|            |-... (Same as Set5)  
|        |-Manga109 (.png)  
|            |-... (Same as Set5)  
|-gene_images (*Note*: This folder will appear after the testing.)
|    |-Set5
|        |-recon
|            |-... (Testing results .png)
|        |-sum.txt
|        |-details.txt
|    |-... (Testing sets)
|-models
|    |-__init__.py  
|    |-method.py  
|    |-module.py  
|-results  
|    |-1  
|    |-4  
|    |-... (Sampling rates)
|-utils 
|    |-__init__.py  
|    |-config.py  
|    |-loader.py  
|-test.py  
|-train.py
|-train.sh
```

## 2. Useage ##  
**1) For training MTC-CSNet.**  

* Put the `BSDS500 (.jpg)` and `VOC2012 (.jpg)` into `./dataset/train/`.  
* e.g. If you want to train MTC-CSNet at sampling rate τ = 0.1 with GPU No.0, please run the following command. The train set will be automatically packaged and our MTC-CSNet will be trained with default parameters (Make sure you have enough GPU RAM):  
```
python train.py --device 0 --rate 0.1
```
* Also you can also run our shell script directly, it will automatically train the model at all sampling rates:  
```
sh train.sh
```
* Your trained models (.pth) will save in the `models` folder, it should contain `info.pth`, `model.pth`, `optimizer.pth` and `log.txt`, respectively represent the information during the training process, trained model parameters, optimizer information, and the reconstruction performance (PSNR, SSIM, LPIPS) of the verification set after one training epoch.  

**2) For testing MTC-CSNet.**  
* Put the `Set5 (.png)`, `Set14 (.png)`, `CBSD68 (.png)` and `Manga109 (.png)` folders into `./dataset/test/`.  
* For example, if you want to test MTC-CSNet at sampling rate τ = 0.1 with GPU No.0, please run:  
```
python test.py --device 0 --rate 0.1
```  
* For convenience of testing, this command will perform image sampling and reconstruction upon `all the test datasets` at `one sampling rate`. This is an example of the test results from the command line:  
```
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
Loading model from: /usr/local/Caskroom/miniconda/base/envs/DL/lib/python3.8/site-packages/lpips/weights/v0.1/alex.pth
r    bird.png
r    butterfly.png
r    head.png
r    woman.png
r    baby.png
g    bird.png
g    butterfly.png
g    head.png
g    woman.png
g    baby.png
b    bird.png
b    butterfly.png
b    head.png
b    woman.png
b    baby.png

Set5 test done.
```
* After that, the results of all tests will be saved to `./gene_images/`. `recon` folder includes all the reconstructed images, `sum.txt` shows the average results of the test set, `detail.txt` shows the Each result of the test set.  
## End ##  
