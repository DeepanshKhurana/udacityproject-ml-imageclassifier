# Imports

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data

from torchvision import datasets, models, transforms

from collections import OrderedDict
import os
import argparse

# Functions

def arg_parser():
    '''
    Takes in command-line arguments and parses them for usage of our Python functions.
    '''
    
    parser = argparse.ArgumentParser(description='ImageClassifier Params')
    
    parser.add_argument('--architecture', 
                        type=str, 
                        help='Architecture and model from torchvision.models as strings: vgg16 and densenet121 supported.')
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Learning Rate for our Neural Network. Default is 0.001.')
    parser.add_argument('--hidden', 
                        type=int, 
                        help='Hidden Units for our Neural Network. Default is 1024.')
    parser.add_argument('--dropout',
                       type=float,
                       help='Dropout value for our Dropout layers. Default is 0.05.')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Epochs for Neural Network training. Default is 1.')
    parser.add_argument('--gpu', 
                        type=str, 
                        help='Use GPU (Y for Yes; N for No). Default is Y.')

    args = parser.parse_args()
    return(args)

def load(data_dir='./flowers'):

    '''
    Loads data for train, test and validation.
    Also loads dataloaders for all three in the same order.
    Returns all six datasets and loaders, in the same order.
    '''

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                               transforms.RandomVerticalFlip(0.5),
                               transforms.RandomRotation(75),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], 
                                                    [0.229, 0.224, 0.225])]),
                   'valid': transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([transforms.RandomResizedCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                   [0.229, 0.224, 0.225])])
                   }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], 64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], 32, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], 32, shuffle=True)
    }

    return(dataloaders['train'], dataloaders['test'], dataloaders['valid'], image_datasets['train'], image_datasets['test'], image_datasets['valid'])

def set_device(gpu):
    '''
    Sets the device based on the parameter. Also handles most edge-cases.
    Returns the device variable to be used later.
    '''
    if gpu=='Y':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device=='cpu':
            print('CUDA not available; using CPU')
        else:
            print('Using GPU')
    elif gpu=='N':
        device = 'cpu'
        print('Using CPU')
    else:
        print('Incorrect Value for GPU entered.')
        print('Fallback to default GPU: 1')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device=='cpu':
            print('CUDA not available; using CPU')
        else:
            print('Using GPU')
    return(device)

def build(device, architecture='vgg16', dropout=0.05, hidden=1024, learning_rate=0.001):
    '''
    Takens in architecture, gpu, dropout, hidden, learning_rate.
    Returns a torch model.
    '''
    if architecture:
        if architecture=='vgg16':
            model = models.vgg16(pretrained=True)
            model.name = architecture
            input_ = 25088
        elif architecture=='densenet121':
            model = models.densenet121(pretrained=True)
            model.name = architecture
            input_ = 1024
        else:
            print('Invalid input: Please use \'vgg16\' or \'densenet121\'')
    else:
        print('No architecture given. Fallback to default architecture: \'vgg16\'')
        model = models.vgg16(pretrained=True)
        model.name = architecture
        input_ = 25088
        
    if hidden:
        hidden = hidden
    else:
        print('No number of hidden inputs specified. Fallback to default inputs: 1024')
        hidden = 1024
        
    if learning_rate:
        learning_rate = learning_rate
    else:
        print('No learning_rate specified. Fallback to default learning_rate: 0.001')
        learning_rate = 0.001
        
    if dropout:
        dropout = dropout
    else:
        print('No dropout specified. Fallback to default dropout: 0.05')
        dropout = 0.05

    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_, hidden)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden, 102, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return(model, criterion, optimizer)

def validation(model, valid_loader, criterion, device):
    '''
    Validation function for our model.
    Returns validation loss and accuracy.
    '''
    valid_loss = 0
    valid_acc = 0
    
    for ii, (inputs, labels) in enumerate(valid_loader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
             
        valid_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        
        equality = (labels.data == ps.max(dim=1)[1])
        valid_acc += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, valid_acc


def train(model, criterion, optimizer, train_loader, valid_loader, device, epochs=1, print_every=50):
    '''
    Trains our Neural Network model
    '''
    steps = 0
    
    if epochs:
        epochs = epochs
    else:
        print('No epochs specified. Fallback to default epochs: 1')
        epochs = 1
    
    print('Training Model for {} epochs'.format(epochs))

    for e in range(epochs):
        
        running_loss = 0
        
        for ii, (inputs, labels) in enumerate(train_loader):
            
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            
            model.zero_grad()
            
            # Forward and backward passes
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, valid_acc = validation(model, valid_loader, criterion, device)
                    
                training_loss = round(float(running_loss/print_every), 3)
                valid_loss = round(float(valid_loss/len(valid_loader)), 3)
                valid_acc = round(float(valid_acc/len(valid_loader)), 3)
                
                print('Epoch: {}/{} :: Training Loss: {} :: Validation Loss: {} :: Validation Accuracy: {}'
                      .format(e+1, epochs, training_loss, valid_loss, valid_acc))
                
                running_loss = 0
                model.train()
    print('Model training complete!')        
    return(model)

def validate(model, test_loader, device):
    '''
    Prints validation accuracy of model
    '''
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = round(100 * correct / total, 2)
    print('Accuracy: {}'.format(accuracy))

def save(model, train_data, epochs, architecture):
    '''
    Saves the model to the given path.
    '''
    model.class_to_idx = train_data.class_to_idx
    
    if epochs:
        epochs = epochs
    else:
        epochs = 1

    checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'epochs': epochs,
                  'architecture': architecture}

    file = 'checkpoint.pth'
    torch.save(checkpoint, file)
    print('Model saved to {}!'.format(file))
    

# Main

def main():

    args = arg_parser()
    if args.gpu:
        gpu=args.gpu
    else:
        print('No input given. Fallback to default GPU: 1')
        gpu='Y'
    device = set_device(gpu)
    train_loader, test_loader, valid_loader, train_data, test_data, valid_data = load()
    model, criterion, optimizer = build(device, architecture=args.architecture, dropout=args.dropout, hidden=args.hidden, learning_rate=args.learning_rate)
    model = train(model=model, train_loader=train_loader, valid_loader=valid_loader, device=device, criterion=criterion, optimizer=optimizer, epochs=args.epochs)
    validate(model=model, test_loader=test_loader, device=device)
    save(model=model, train_data=train_data, epochs=args.epochs, architecture = args.architecture)

if __name__ == '__main__':
    main()
