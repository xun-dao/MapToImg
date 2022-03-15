# *_*coding:utf-8 *_*
"""
@author: mingruisu
@time: 2022/3/15 11:01 AM
@desc:
"""
from ..models.EdgeModel import EdgeModel
import os
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import random
import scipy
import torch
from skimage.feature import canny
from scipy.misc import imread
from skimage.color import rgb2gray, gray2rgb
from ..models.InpaintingModel import InpaintingModel
# 加载模型

# 获取数据

# 执行推理

# 返回结果

def forward(images_gray, edges, masks, images, edge_model, inpaint_model, results_path, result_img_name):

    edges = edge_model(images_gray, edges, masks).detach()
    outputs = inpaint_model(images, edges, masks)
    outputs_merged = (outputs * masks) + (images * (1 - masks))

    output = postprocess(outputs_merged)[0]
    path = os.path.join(results_path, result_img_name)

    im = Image.fromarray(output.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)

    return path


def get_data(size, img_path, mask_path, sigma=2):
    # size = self.input_size

    # load image
    img = imread(img_path)

    # gray to rgb
    if len(img.shape) < 3:
        img = gray2rgb(img)

    # resize/crop if needed
    if size != 0:
        img = resize(img, size, size)

    # create grayscale image
    img_gray = rgb2gray(img)

    # load mask
    imgh, imgw = img.shape[0:2]
    mask = load_mask(imgh, imgw, mask_path)

    # load edge
    edge = load_edge(img_gray, mask, sigma)

    # augment data
    if np.random.binomial(1, 0.5) > 0:
        img = img[:, ::-1, ...]
        img_gray = img_gray[:, ::-1, ...]
        edge = edge[:, ::-1, ...]
        mask = mask[:, ::-1, ...]

    return to_tensor(img), to_tensor(img_gray), to_tensor(edge), to_tensor(mask)


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()

    return torch.stack([img_t], dim=0)

def resize(img, height, width, centerCrop=True):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    img = scipy.misc.imresize(img, [height, width])
    return img

def load_mask(imgh, imgw, mask_img_path):
    # test mode: load mask non random
    mask = imread(mask_img_path)
    mask = resize(mask, imgh, imgw, centerCrop=False)
    mask = rgb2gray(mask)
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


def load_edge(img, mask, sigma=2):

    # in test mode images are masked (with masked regions),
    # using 'mask' parameter prevents canny to detect edges for the masked regions
    mask = (1 - mask / 255).astype(np.bool)

    if sigma == -1:
        return np.zeros(img.shape).astype(np.float)

    # random sigma
    if sigma == 0:
        sigma = random.randint(1, 4)

    return canny(img, sigma=sigma, mask=mask).astype(np.float)


def load_model(config):
    edge_model = EdgeModel(config).to(config.DEVICE)
    inpaint_model = InpaintingModel(config).to(config.DEVICE)

    edge_model.load()
    inpaint_model.load()

    edge_model.eval()
    inpaint_model.eval()
    return edge_model, inpaint_model


def cuda(self, *args):
    return (item.to(self.config.DEVICE) for item in args)

def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

