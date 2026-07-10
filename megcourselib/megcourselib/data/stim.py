import os
from os import path

import imageio
import numpy as np
import torch
from PIL import Image

from megcourselib.constants import SCREEN_H, SCREEN_W
from megcourselib.data.util import get_scene_path


def extract_crop(
    scene_im: Image.Image,
    gx: float,
    gy: float,
    crop_w: int,
    crop_h: int,
) -> tuple:
    """Extract a crop centred on a fixation point.

    Parameters
    ----------
    scene_im : PIL Image, already scaled to presentation size
    gx, gy   : fixation coordinates in screen-centred pixels
                (origin = screen centre, y-axis up)
    crop_w, crop_h : output crop size in pixels

    Returns
    -------
    crop  : np.ndarray uint8 (crop_h, crop_w, 3), zero array if out of bounds
    valid : bool
    """
    im_w, im_h = scene_im.size

    # Convert screen-centred coordinates to image-space (top-left origin)
    left = (gx - SCREEN_W / 2) + im_w / 2 - crop_w / 2
    top = im_h / 2 - (gy - SCREEN_H / 2) - crop_h / 2
    right = left + crop_w
    bottom = top + crop_h

    if left < 0 or top < 0 or right > im_w or bottom > im_h:
        return np.zeros((crop_h, crop_w, 3), dtype=np.uint8), False

    crop = scene_im.crop((left, top, right, bottom))
    return np.array(crop, dtype=np.uint8), True


def load_crop(scene_id, gx, gy, cropsize, stim_root):
    im = Image.open(get_scene_path(scene_id, stim_root)).convert("RGB")
    crop, is_valid = extract_crop(im, gx, gy, crop_w=cropsize[0], crop_h=cropsize[1])
    return is_valid, crop


def load_image_folder_as_tensors(imfolder):
    """
    Given a folder of images, load all images and return them
    as a tensor sorted alphabetically by filename.
    The returned tensor will be a torch tensor of dtype float (values in [0..1])
    The tensor will have shape:
        [n_images, channels, width, height]
    Usually, images will be loaded as rgb, so channels = 3
    """
    imnames = np.sort(os.listdir(imfolder))
    images = list()
    for fname in imnames:
        images.append(imageio.v2.imread(path.join(imfolder, fname)))
    images = torch.from_numpy(np.array(images).astype(np.float32) / 255.0)
    images = torch.moveaxis(images, -1, 1)
    return images, imnames
