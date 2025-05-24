import numpy as np
from monai.transforms import RandFlip
import random
import torch


class RandomFlipR():
    def __init__(self, prob):
        self.prob = prob
        self.flipers = []
        self.flipers.append(RandFlip(spatial_axis=0, prob=1))
        self.flipers.append(RandFlip(spatial_axis=1, prob=1))
        self.flipers.append(RandFlip(spatial_axis=2, prob=1))

    def __call__(self, img: torch.Tensor, *args, **kwargs):
        flip_count = 0
        for i in range(3):
            n = random.random()
            if n < self.prob:
                img = self.flipers[i](img)
                flip_count += 1
        return img, flip_count



    # def __call__(self, img: torch.Tensor, *args, **kwargs):
    #     flip_count = 0
    #
    #     img = self.flipers[0](img)
    #     return img, flip_count
class RandomFlipR_v2():
    def __init__(self, prob):
        self.prob = prob
        self.flipers = []
        self.flipers.append(RandFlip(spatial_axis=0, prob=1))
        self.flipers.append(RandFlip(spatial_axis=1, prob=1))
        self.flipers.append(RandFlip(spatial_axis=2, prob=1))

    def __call__(self, img: torch.Tensor, *args, **kwargs):
        flip_count = 0
        for i in range(3):
            n = random.random()
            if n < self.prob:
                img = self.flipers[i](img)
                if i == 2:
                    flip_count = 1
        return img, flip_count

if __name__=="__main__":
    pass
