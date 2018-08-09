# coding=utf-8
"""
Train.py, to train and save model for flower image data set 
"""

import argparse
from image_classifier_project import *

def main():

    in_arg = parse_input_args()

    # Set the default hyperparameters if none given
    if in_arg.hidden_units==[]:
        in_arg.hidden_units=[4096, 4096];

    # Create the save file path name
    save_path = in_arg.save_dir + in_arg.checkpoint

    # Set the inputs to the hyperparameters
    #users to set hyperparameters for learning rate, number of hidden units, and training epochs
    hyperparameters = {'learnrate': in_arg.learning_rate,
                       'hidden_layers': in_arg.hidden_units,
                       'epochs': in_arg.epochs,
                       'architecture': in_arg.arch,
                       'dropout_probability': in_arg.drop_p}

    # Get the transfer model (vgg11, vgg13, vgg16, vgg19_bn and alexnet)
    model = get_model(hyperparameters['architecture'])

    # Create the dataloaders for training and testing images
    # Also, create the classifier based on the given inputs and attach it to the transfer model
    model, train_dataloader, test_dataloader = model_config(in_arg.data_dir, 
                                                            model,  
                                                            hyperparameters['hidden_layers'],
                                                            hyperparameters['dropout_probability'])

    if in_arg.gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    # Print the relevant parameters to the output window before training
    print("\n==============================================================\n")
    print("Training data:                {}".format(in_arg.data_dir + '/train'))
    print("Validation data:              {}".format(in_arg.data_dir + '/valid'))
    print("Testing data:              {}".format(in_arg.data_dir + '/test'))
    print("Checkpoint will be saved to:  {}".format(save_path))
    print("Device:                       {}".format(device))

    print("Transfer model:               {}".format(hyperparameters['architecture']))
    print("Hidden layers:                {}".format(hyperparameters['hidden_layers']))
    print("Learning rate:                {}".format(hyperparameters['learnrate']))
    print("Dropout probability:          {}".format(hyperparameters['dropout_probability']))
    print("Epochs:                       {}".format(hyperparameters['epochs']))
    print("\n==============================================================\n")
    
    model_accuracy = train(model, train_dataloader, test_dataloader, device, 
                           hyperparameters['learnrate'], 
                           epochs=hyperparameters['epochs'], 
                           print_every=40, 
                           debug=0)

    # Save the model checkpoint
    checkpoint_save(save_path, model, hyperparameters['architecture'], model.classifier.hidden_layers[0].in_features, 
                    model.classifier.output.out_features, hyperparameters, model_accuracy)

# End of Main

def parse_input_args():

    """
    parse_input_args() to parse command line arguments.
    
    usage: train.py [-h] [--save_dir SAVE_DIR] [--checkpoint CHECKPOINT]
                [--arch ARCH] [--learning_rate LEARNING_RATE]
                [--hidden_units HIDDEN_UNITS] [--epochs EPOCHS]
                [--drop_p DROP_P] [--gpu]
                data_dir

    positional arguments:
      data_dir              full path name to data directory

    optional arguments:
      -h, --help            show this help message and exit
      --save_dir SAVE_DIR   full path to save model checkpoints
      --checkpoint CHECKPOINT
                            name of checkpoint file to save
      --arch ARCH           chosen model options are: alexnet, vgg11, vgg13,
                            vgg16, vgg19_bn; Default: vgg16
      --learning_rate LEARNING_RATE
                            learning rate; float; Default: 0.001
      --hidden_units HIDDEN_UNITS
                            hidden layer unit, call multiple times to add hidden
                            units
      --epochs EPOCHS       number of epochs
      --drop_p DROP_P       dropout probability
      --gpu                 use GPU instead of CPU
    """

    #Read https://pymotw.com/3/argparse/ for more details.

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action="store", type=str, default='./flowers/',
                        help='full path name to data directory')
    
    parser.add_argument('--save_dir', type=str, default='./model_check_points/', 
                        help='full path to save model checkpoints')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', 
                        help='name of checkpoint file to save')
    parser.add_argument('--arch', type=str, default='vgg16', 
                        help='chosen model options are: alexnet, vgg11, vgg13, vgg16, vgg19_bn; Default: vgg16')
    
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='learning rate; float; Default: 0.001')
    parser.add_argument('--hidden_units', action="append", type=int, default=[], 
                        help='hidden layer unit, call multiple times to add hidden units')
    
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--drop_p', type=float, default=0.2, help='dropout probability')

    parser.add_argument('--gpu', action="store_true", default=False, 
                        help='use GPU instead of CPU')
    return parser.parse_args()
#End of Parse_input_args()

if __name__ == '__main__':
    main()
