# Imports here
import os
import argparse
import util
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import numpy as np

from workspace_utils import active_session

from pathlib import Path

def load_data(data_directory):
    # Load the data
    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Knobs and switches
    dataloader_batch_size = 64

    # ImageNet normalization values
    mean_norm = util.MEAN_NORM_IMAGENET
    std_norm = util.STD_NORM_IMAGENET
    
    normalize_xform = transforms.Normalize(mean_norm, std_norm)

    resize_crop_only_xform = transforms.Compose([
        transforms.Resize(util.TARGET_IMAGE_SIZE),
        transforms.CenterCrop(util.CENTER_CROP_SIZE),
        transforms.ToTensor(),
        normalize_xform
    ])

    # For training set, apply random transformations like cropping, scaling, rotating
    # For validation and test sets, apply no transformations except for resize and crop.

    data_transforms = {
        'training' : transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(util.CENTER_CROP_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_xform]),
        'validation' : resize_crop_only_xform,
        'test' : resize_crop_only_xform
    }

    image_datasets = {
        'training' : datasets.ImageFolder(
            train_dir, transform = data_transforms['training']),
        'validation' : datasets.ImageFolder(
            valid_dir, transform = data_transforms['validation']),
        'test' : datasets.ImageFolder(
            test_dir, transform = data_transforms['test'])
    }

    dataloaders = {
        'training' : DataLoader(
            image_datasets['training'], 
            batch_size=dataloader_batch_size, shuffle=True),
        'validation' : DataLoader(
            image_datasets['validation'], 
            batch_size=dataloader_batch_size, shuffle=True),
        'test' : DataLoader(
            image_datasets['test'], 
            batch_size=dataloader_batch_size, shuffle=True)
    }
    
    return image_datasets, dataloaders


def build_model(arch, learning_rate, hidden_units, dropout_rate):
    # BUILD

    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        print('Architecture: VGG19')
        input_units = util.QTY_VGG19_INPUT_UNITS
        hidden_0_units = util.QTY_VGG19_HIDDEN_0_UNITS
    elif arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        print('Architecture: VGG11')
        input_units = util.QTY_VGG11_INPUT_UNITS
        hidden_0_units = util.QTY_VGG11_HIDDEN_0_UNITS
    else:
        arch = 'vgg19'
        model = models.vgg19(pretrained=True)
        print('Architecture: defaulted to VGG19')
        input_units = util.QTY_VGG19_INPUT_UNITS
        hidden_0_units = util.QTY_VGG19_HIDDEN_0_UNITS

    # Freeze features
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier_struct = OrderedDict([
        ('lin1', nn.Linear(input_units, hidden_0_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(dropout_rate)),
        ('lin2', nn.Linear(hidden_0_units, hidden_units)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(dropout_rate)),
        ('lin3', nn.Linear(hidden_units, util.QTY_CATEGORIES)),
        ('output', nn.LogSoftmax(dim=1))
    ])
    print(f'{hidden_units} hidden units')
    classifier = nn.Sequential(classifier_struct)

    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer, classifier_struct, arch

def train(device, epochs, model, criterion, optimizer, dataloaders):
    # TRAIN
    steps = 0
    running_loss = 0
    running_loss_steps = 0
    print_every = 50
    model.to(device)
    print(f"Beginning training with {len(dataloaders['training'])} images and {epochs} epochs on '{device}'\n")

    with active_session():
        model.train()
        for epoch in range(epochs):
            for inputs, labels in dataloaders['training']:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_loss_steps += 1

                step_of_epoch = steps % len(dataloaders['training'])
                if steps % print_every == 0 or step_of_epoch == 0:
                    test_loss = 0
                    accuracy_sum = 0
                    qty_processed_inputs = 0
                    model.eval()
                    with torch.no_grad():
                        # One iteration per batch
                        for inputs, labels in dataloaders['validation']:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            qty_processed_inputs += len(equals)
                            accuracy_sum += torch.sum(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}, after "
                          f"training round {len(dataloaders['training']) if step_of_epoch == 0 else step_of_epoch}/{len(dataloaders['training'])}.. "
                          f"Train loss: {(running_loss/running_loss_steps):.3f}.. "
                          f"Validation loss: {test_loss/len(dataloaders['validation']):.3f}.. "
                          f"Validation accuracy: {accuracy_sum/qty_processed_inputs:.3f}")
                    running_loss = running_loss_steps = 0
                    model.train()

    print('Training complete!')
    return model, optimizer

def save_checkpoint(save_dir, model, classifier_struct, image_datasets, arch):
    # Assumes training is complete and model is ready for inference
    checkpoint = {'classifier': classifier_struct,
                  'class_to_idx': image_datasets['training'].class_to_idx,
                  'model_state_dict': model.state_dict(),
                  'pretrained_type': arch
                 }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')
    print(f'Checkpoint saved to {save_dir}/checkpoint.pth')
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', action="store")
    parser.add_argument('--save_dir', default='./', action="store", 
                        help='Path to which checkpoint of trained model will be saved')
    parser.add_argument('--arch', default='vgg19', action="store", 
                        help='Pretrained model architecture. Current valid values are "vgg11" and "vgg19".')
    parser.add_argument('--learning_rate', action="store",
                        type=float, default='0.001', 
                        help='Learning rate for training model')
    parser.add_argument('--hidden_units', action="store",
                        type=int, default='1024', 
                        help='Number of units in hidden layer before output layer')
    parser.add_argument('--dropout_rate', action="store",
                        type=float, default='0.2', 
                        help='Training dropout rate')
    parser.add_argument('--epochs', action="store",
                        type=int, default='2', 
                        help='Training epochs')
    parser.add_argument('--gpu', action="store_true",
                        default=False, 
                        help='Use GPU for training if available. If false or GPU is not available, CPU will be used.')

    args = parser.parse_args()
          
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    
    print('Loading image data...')
    image_datasets, dataloaders = load_data(args.data_directory)
    print('Building model...')
    model, criterion, optimizer, classifier_struct, arch = build_model(args.arch, args.learning_rate, 
                                                                       args.hidden_units, args.dropout_rate)
    print(f'Training model...')
    model, optimizer = train(device, args.epochs, model, criterion, optimizer, dataloaders)
    print('Saving checkpoint...')
    save_checkpoint(args.save_dir, model, classifier_struct, image_datasets, arch)
    print()
    
if __name__ == '__main__':
    main()

