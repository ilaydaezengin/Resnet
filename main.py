import numpy as np
from PIL import Image,ImageEnhance
from resnet import resnet_model
import tensorflow as tf


def main():
    t = resnet_model(120,224,224,64,0.01)
    t.run(10)

main()
