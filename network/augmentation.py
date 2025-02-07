import torch, sys
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import importlib
from my_utils import constants
from my_utils.constants import Const_c
# Initialize reading the json constants file for each experiment
exp = str(sys.argv[2])
full_name = str(sys.argv[3])
Constants_c = Const_c(exp, full_name)
Constants = Constants_c.Constants


class AugmentStage(nn.Module):
    def __init__(self):
        super(AugmentStage, self).__init__()
        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    Constants["INPUT_SIZE"][list(Constants['TGT_DATASETS'].keys())[0]],
                    scale=(0.5, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23)], p=0.5
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack([self.transforms(image) for image in x])
