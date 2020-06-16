# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Training 
Current valid architectures implemented in this project are 'vgg11' and 'vgg19'

## Parameter descriptions 
```
# python train.py -h
usage: train.py [-h] [--save_dir SAVE_DIR] [--arch ARCH]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--dropout_rate DROPOUT_RATE] [--epochs EPOCHS] [--gpu]
                data_directory

positional arguments:
  data_directory

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Path to which checkpoint of trained model will be
                        saved
  --arch ARCH           Pretrained model architecture. Current valid values
                        are "vgg11" and "vgg19".
  --learning_rate LEARNING_RATE
                        Learning rate for training model
  --hidden_units HIDDEN_UNITS
                        Number of units in hidden layer before output layer
  --dropout_rate DROPOUT_RATE
                        Training dropout rate
  --epochs EPOCHS       Training epochs
  --gpu                 Use GPU for training if available. If false or GPU is
                        not available, CPU will be used.
```
## Usage examples
Usage examples can be found in run_train.sh

# Inference

## Parameter descriptions
```
# python predict.py -h
usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu]
                  image_path checkpoint_path

positional arguments:
  image_path            Path to image file to be inferred by trained model
  checkpoint_path       Path of checkpoint file for trained model

optional arguments:
  -h, --help            show this help message and exit
  --top_k TOP_K         Number of top k class results to be displayed, per
                        probability
  --category_names CATEGORY_NAMES
                        Mapping of classes (categories) to actual names for
                        displaying results
  --gpu                 Use GPU for inference if available. If false or GPU is
                        not available, CPU will be used.
```

## Usage examples
Usage examples can be found in run_predict.sh

# Notes
workspace_utils.py is from https://github.com/udacity/workspaces-student-support.
Course materials at https://github.com/udacity/AIPND were used for reference during this project.
