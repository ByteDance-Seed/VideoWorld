# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np


def gaussian(x, sigma):
    """Gaussian function.

    Args:
        x (array_like): The independent variable.
        sigma (float): Standard deviation of the gaussian function.

    Return:
        np.ndarray or scalar: Gaussian value of `x`.
    """
    return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def dgaussian(x, sigma):
    """Gradient of gaussian.

    Args:
        x (array_like): The independent variable.
        sigma (float): Standard deviation of the gaussian function.

    Return:
        np.ndarray or scalar: Gradient of gaussian of `x`.
    """
    return -x * gaussian(x, sigma) / sigma**2


def gauss_filter(sigma, epsilon=1e-2):
    """Gradient of gaussian.

    Args:
        sigma (float): Standard deviation of the gaussian kernel.
        epsilon (float): Small value used when calculating kernel size.
            Default: 1e-2.

    Return:
        filter_x (np.ndarray): Gaussian filter along x axis.
        filter_y (np.ndarray): Gaussian filter along y axis.
    """
    half_size = np.ceil(
        sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
    size = int(2 * half_size + 1)

    # create filter in x axis
    filter_x = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            filter_x[i, j] = gaussian(i - half_size, sigma) * dgaussian(
                j - half_size, sigma)

    # normalize filter
    norm = np.sqrt((filter_x**2).sum())
    filter_x = filter_x / norm
    filter_y = np.transpose(filter_x)

    return filter_x, filter_y


def gauss_gradient(img, sigma):
    """Gaussian gradient.

    From https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/
    submissions/8060/versions/2/previews/gaussgradient/gaussgradient.m/
    index.html

    Args:
        img (np.ndarray): Input image.
        sigma (float): Standard deviation of the gaussian kernel.

    Return:
        np.ndarray: Gaussian gradient of input `img`.
    """
    filter_x, filter_y = gauss_filter(sigma)
    img_filtered_x = cv2.filter2D(
        img, -1, filter_x, borderType=cv2.BORDER_REPLICATE)
    img_filtered_y = cv2.filter2D(
        img, -1, filter_y, borderType=cv2.BORDER_REPLICATE)
    return np.sqrt(img_filtered_x**2 + img_filtered_y**2)
