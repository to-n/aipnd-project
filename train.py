# Imports here
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import numpy as np

from workspace_utils import active_session

from pathlib import Path




# Load the data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Knobs and switches
dataloader_batch_size = 64
qty_output_units = 102
learn_rate = 0.001
epochs = 2

# ImageNet normalization values
mean_norm = [0.485, 0.456, 0.406]
std_norm = [0.229, 0.224, 0.225]

normalize_xform = transforms.Normalize(mean_norm, std_norm)

resize_crop_only_xform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize_xform
])

# For training set, apply random transformations like cropping, scaling, rotating
# For validation and test sets, apply no transformations except for resize and crop.

data_transforms = {
    'training' : transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
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


# BUILD
model = models.vgg19(pretrained=True)

# Freeze features
for param in model.parameters():
    param.requires_grad = False
    
from collections import OrderedDict
classifier_struct = OrderedDict([
    ('fc1', nn.Linear(25088, 4096)),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(0.2)),
    ('fc2', nn.Linear(4096, 1024)),
    ('relu2', nn.ReLU()),
    ('dropout2', nn.Dropout(0.2)),
    ('fc3', nn.Linear(1024, qty_output_units)),
    ('output', nn.LogSoftmax(dim=1))
])

classifier = nn.Sequential(classifier_struct)

model.classifier = classifier

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# TRAIN
steps = 0
running_loss = 0
print_every = 50

print(f"Beginning training with {len(dataloaders['training'])}\
 images and {epochs} epochs\n")

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
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(dataloaders['validation']):.3f}.. "
                      f"Validation accuracy: {accuracy_sum/qty_processed_inputs:.3f}")
                running_loss = 0
                model.train()

print('Training complete!')

# Assumes training is complete and model is ready for inference
checkpoint = {'classifier': classifier_struct,
              'class_to_idx': image_datasets['training'].class_to_idx,
              'model_state_dict': model.state_dict(),
              'pretrained_type': 'vgg19'
             }

torch.save(checkpoint, 'checkpoint.pth')

