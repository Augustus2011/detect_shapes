import numpy as np
import torch
import cairo
import cv2
import pandas as pd
import matplotlib
import ultralytics
import tqdm
import sklearn


class set_seed:
    def __init__(self,seed:int=42):
        self.seed=seed
    def set(self)->None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

def check_dependencies()->None:
    print("check dependencies version")
    print("numpy:",np.__version__)
    print("torch",torch.__version__)
    print("cairo",cairo.version)
    print("open-cv",cv2.__version__)
    print("pandas",pd.__version__)
    print("matplotlib",matplotlib.__version__)
    print("ultralytics",ultralytics.__version__)
    print("tqdm",tqdm.__version__)
    print("sklearn",sklearn.__version__)
    print("-"*20)


def check_return_device()->str:
    device='cuda'if torch.cuda.is_available()else'cpu'
    if device=='cuda':
        print("numbers:",torch.cuda.device_count())
        print("current device:",torch.cuda.current_device())
        print("device name:",torch.cuda.get_device_name(0))
        print("-"*20,"🤗")
    else:
        print("got cpu 😢")
    return device
