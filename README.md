## The pytorch implementation of DCGAN

This code add residual block and simplified the source code. If you want to visualize the training process, you need to type `python -m visdom.server` to opeh visdom.

Since it have residual block, you best to augment data by `data_augment.py`, which can get better results.

Paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversial Networks](https://arxiv.org/abs/1511.06434)

data: 5 w anime images dataset [80k/13GB][baidu pandownload password: g5qa](https://pan.baidu.com/s/1eSifHcA)

## Requirement

```
pip install -r requirements.txt
```



## Usage

Training model

```
python main.py train --gpu=True --vis=True
```

Test examples:

```
python main.py generate --vis=False
```

complete parameters:

```python
data = './data/'          # the road to store data
ndf = 128       # the channel of the first convolutional layer of the discriminator net 
ngf = 128       # the channel of the last convolutional layer of the generator net
batch_size = 256
img_size = 64
max_epoch = 50            # numbers of iterations
lr = 1e-4                 # learning rate
beta = 0.5                # Adam optimizer beat_1
nz = 100                  # the channel of noise  100 x 1 x 1
gpu = True                # Use GPU
d_every = 1               # Train discriminator every 1 batch
g_every = 5               # Train generator every 5 batch
vis = True                # Use visdom
plot_every = 20           # Visdom plot every 20 batch 
net_path = './checkpoints'

gen_num = 512             # generate 512 images
gen_select_num = 64       # select best 64 images
gen_img = 'result.png'
gen_mean = 0
gen_std = 1
```



## Result



