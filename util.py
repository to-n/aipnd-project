from PIL import Image
import numpy as np

import json

MEAN_NORM_IMAGENET = [0.485, 0.456, 0.406]
STD_NORM_IMAGENET = [0.229, 0.224, 0.225]
TARGET_IMAGE_SIZE = 255
CENTER_CROP_SIZE = 224

QTY_VGG19_INPUT_UNITS = 25088
QTY_VGG19_HIDDEN_0_UNITS = 4096
QTY_VGG11_INPUT_UNITS = 25088
QTY_VGG11_HIDDEN_0_UNITS = 4096
QTY_CATEGORIES = 102

def load_class_name_map(map_path):
    '''Load JSON mappping of classes to names
        '''
    with open(map_path, 'r') as f:
        return json.load(f)
    
def process_image(image_path):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns a numpy array
        '''
    image = Image.open(image_path)
    # Resize and crop
    width, height = image.size
    old_short, old_long = min([width, height]), max([width, height])
    new_short = 256
    new_long = int(new_short*old_long/old_short)
    new_size = [new_short, new_long] if old_short == width else [new_long, new_short]
    image = image.resize(new_size)

    center_size = 224
    width, height = image.size
    left = (width - center_size)/2
    right = (width + center_size)/2
    top = (height - center_size)/2
    bottom = (height + center_size)/2
    image = image.crop((left, top, right, bottom))
    width, height = image.size

    # Normalize
    np_image = np.array(image)/255.0
    np_image -= [0.485, 0.456, 0.406]
    np_image /= [0.229, 0.224, 0.225]

    # Transpose
    return np_image.transpose(2, 0, 1)
