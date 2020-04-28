# Imports

import pandas as pd
import numpy as np

import torch
import json
import PIL
from torchvision import models, transforms

from train import set_device

import os
import argparse
import sys

# Functions

def arg_parser():
    '''
    Takes in command-line arguments and parses them for usage of our Python functions.
    '''
    
    parser = argparse.ArgumentParser(description='ImageClassifier Prediction Params')
    
    parser.add_argument('--gpu', 
                        type=str, 
                        help='Use GPU (Y for Yes; N for No). Default is Y.')

    parser.add_argument('--checkpoint',
                        type=str,
                        help='Path for model checkpoint created using train.py. Default is \'./checkpoint.pth\'.')

    parser.add_argument('--image',
                        type=str,
                        help='Path for image to be predicted. If not included, the script will crash.')

    parser.add_argument('--catmap',
                        type=str,
                        help='Path to Category Map JSON file. If not included, integers will be shown.')

    parser.add_argument('--topk', 
                        type=int, 
                        help='Top K predictions to show. Default is 5.')

    args = parser.parse_args()

    return(args)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    loaded_image = PIL.Image.open(image)
    
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return(transformations(loaded_image))

def load_model(checkpoint, device):
    
    checkpoint = torch.load(checkpoint)
    architecture = checkpoint['architecture']
    
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        model = model.to(device)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        model = model.to(device)
    else:
        print('Unsupported architecture. Only \'vgg16\' and \'densenet121\' are allowed. \nPlease use train.py to create the model.')
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return(model)

def predict(model, image, device, catmap, k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = image.unsqueeze_(0)
    
    model.eval()
    with torch.no_grad():
        output = model.forward(image.to(device))
        
    probabilities = torch.exp(output)

    if k:
        k = k
    else:
        print('No Top K specified. Fallback to default Top K: 5')
        k = 5
        
    topk_probabilities, topk_labels = probabilities.topk(k)
    
    labels_list = topk_labels.squeeze().tolist()
    probabilities_list = topk_probabilities.squeeze().tolist()
    probabilities_list = [round(prob, 3) for prob in probabilities_list]

    if catmap:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        text_labels_list = []
        for label in labels_list:
            text_labels_list.append(cat_to_name[str(label)])
        labels_list = text_labels_list
    else:
        final_dict = dict(zip(labels_list, probabilities_list))
        final_dict = sorted(final_dict.items(), key=lambda kv: kv[1], reverse=True)
        
    final_dict = dict(zip(labels_list, probabilities_list))
    
    return(final_dict)

# Main

def main():
    
    args = arg_parser()
    if args.gpu:
        gpu=args.gpu
    else:
        print('No input given. Fallback to default GPU: 1')
        gpu='Y'
    device = set_device(gpu)
    if args.checkpoint:
        model = load_model(args.checkpoint, device)
    else:
        model = load_model('checkpoint.pth', device)
    if args.image:
        image = process_image(args.image)
    else:
        print('No image path provided. Exiting.')
        sys.exit()
    predictions = predict(model, image, device, args.catmap, args.topk)
    print(predictions)

if __name__ == '__main__':
    main()
