# Project name: Extensions to Generative Adversarial Neural Networks for Image Generation

## Desciption
This is a final project developed by Jim Jiayi Xu, Zhao Binglin, Xiufeng Zhao, Xiaoyin Yang, Liang Hou for UCSD Fall 2018 ee285 course.  
Contact us if you have any problem:{jjx002, bzhao, x6zhao, x4yang, l7hou}@eng.ucsd.edu

## Prerequisites
We use pytorch and colab/jupyternotebook as ...
Below is What you need to install:
```
# http://pytorch.org/
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
```
 
## 

 
 
## User Instruction
