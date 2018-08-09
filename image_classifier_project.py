from collections import OrderedDict

import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import os, random
from PIL import Image
from math import ceil


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
#End of Network Class
        
def model_config(data_dir, model, hidden_layers, dropout_probability):

    training_dir = data_dir + '/train'
    validation_dir = data_dir + '/valid'
    testing_dir = data_dir + '/test'

    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #4
    #Data batching
    #The data for each set is loaded with torchvision's DataLoader

    # Done: Load the datasets with ImageFolder
    #image_datasets for training validation and testing
    training_set = datasets.ImageFolder(training_dir, transform = training_transforms)
    validation_set = datasets.ImageFolder(validation_dir, transform = validation_transforms)
    testing_set = datasets.ImageFolder(testing_dir, transform = testing_transforms)

    #5
    #Data loading
    #The data for each set (train, validation, test) is loaded with torchvision's ImageFolder

    # Done: Using the image datasets and the trainforms, define the dataloaders
    #dataloaders for training validation and testing
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32,shuffle=True)
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=32,shuffle=True)

    '''
    # Create the classifier for the model 
    # Use the transfer models outputs as inputs and the number of image categories as outputs
    classifier = Network(model.classifier[0].in_features,102,hidden_layers, dropout_probability)
    '''
    '''
    model.classifier[0].in_features is not working with alexnet... generic solution is below
    '''
    classifier_modules = [type(model.classifier[i]).__name__ for i in range(len(model.classifier))]
    First_Linear_Module = classifier_modules.index('Linear')

    # Create the classifier for the model 
    # Use the transfer models outputs as inputs and the number of image categories as outputs
    #classifier = Network(model.classifier[First_Linear_Module].in_features, len(train_dataset.classes), hidden_layers, dropout_probability)
    classifier = Network(model.classifier[First_Linear_Module].in_features,102,hidden_layers, dropout_probability)
    # Attach the classifier to the model
    model.classifier = classifier

    return model, training_loader, testing_loader
#end of model_config        
        
def train(model, trainloader, testloader, device, learning_rate, epochs=5, print_every=40, debug=0):
    
#    device = torch.device(device)
   
    #Setup training Parameters
    steps = 0
    running_loss = 0
    
    if(debug == 1):
        print("Epochs | Training Loss | Test Loss | Test Accuracy ")
    # Define the criterion and optimizer with the model and learning rate (0.001)
    criterion = nn.NLLLoss().to(device)
    # Only train the classifier parameters, feature parameters are frozen
    #optimizer = optim.Adam(list(model.classifier.parameters()), learning_rate)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Send the model to the correct processor
    model.to(device)
        
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        
        for images, labels in trainloader:
            steps += 1
            
            images, labels = images.to(device), labels.to(device)            
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion,device)
                if(debug == 1):
                    print("{}/{} |".format(e+1, epochs),
                          "{:.3f} |".format(running_loss/print_every),
                          "{:.3f} | ".format(test_loss/len(testloader)),
                          "{:.3f} |".format(accuracy/len(testloader)))
                else:
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()
                
                
    #To print final Test & Running loss& Test Accuracy
    
    print("\nFinal Test & Running loss & Test Accuracy\n-----\n")
    model.eval()

    # Turn off gradients for validation, will speed up inference
    with torch.no_grad():
        test_loss, accuracy = validation(model, testloader, criterion,device)
    if(debug == 1):
        print("{}/{} |".format(e+1, epochs),
              "{:.3f} |".format(running_loss/print_every),
              "{:.3f} | ".format(test_loss/len(testloader)),
              "{:.3f} |".format(accuracy/len(testloader)))
    else:
        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/print_every),
              "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

    # Return the accuracy in terms of a percentage
    return accuracy/len(testloader)
#End of train
    
def validation(model, testloader, criterion, device = 'cpu'):
    accuracy = 0
    test_loss = 0
    # Send the model to the desired processor
    model.to(device)

    for images, labels in testloader:

        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy
#End of Validation
    
# This function is to make generic implementation to load torchvision models
def get_model(model_name):
    if model_name=='alexnet':
        model = models.alexnet(pretrained=True)
    elif model_name=='vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
    elif model_name=='vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name=='vgg13':
        model = models.vgg13(pretrained=True)
    elif model_name=='vgg11':
        model = models.vgg11(pretrained=True)
    else:
        print("\nError: selecting default vgg16\n")
        model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    return model  
#End of get_model
    
def checkpoint_save(filepath, model, transfer_model_name, input_size, output_size, hyperparams, model_accuracy):
    checkpoint = {'transfer_model_name': transfer_model_name,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hyperparams['hidden_layers'],
                  'drop_p': hyperparams['dropout_probability'],
                  'learnrate': hyperparams['learnrate'],
                  'epochs': hyperparams['epochs'],
                  'model_accuracy': model_accuracy,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, filepath)         
#End of checkpoint_save

def checkpoint_load(filepath):
    checkpoint = torch.load(filepath)
    model_name = checkpoint['transfer_model_name']
    

    model = get_model(model_name)
    
    classifier = Network(checkpoint['input_size'],
                         checkpoint['output_size'],
                         checkpoint['hidden_layers'],
                         checkpoint['drop_p'])
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    hidden_layers = checkpoint['hidden_layers']
    dropout_probability = checkpoint['drop_p']
    learnrate = checkpoint['learnrate']
    epochs = checkpoint['epochs']
    accuracy = checkpoint['model_accuracy']
    
    return model, accuracy, learnrate
#End of checkpoint_load

def process_image(pil_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Done: Process a PIL image for use in a PyTorch model
    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(pil_image)
    
    # add dimension for batch
    img_tensor.unsqueeze_(0)
    
    return img_tensor
# End of process_image
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
#End of imshow
    
def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(Image.open(image_path))
    
    model.eval()
    model.to(device)
    criterion = nn.NLLLoss()
#    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
    image = image.to(device)
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)

    topk_probs_tensor, topk_idx_tensor = ps.topk(topk)

   
    return topk_probs_tensor, topk_idx_tensor
#End of predict    
    
    
                        
