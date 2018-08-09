"""
predict.py This script predicts the category of an image.
"""

import argparse
from image_classifier_project import *
from PIL import Image
import matplotlib.pyplot as plt
import json

def parse_input_args():
    """
    parse_input_args() to parse command line arguments.
    
    usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu] [--input_category INPUT_CATEGORY]
                  input checkpoint

    positional arguments:
      input                 full path name to input image file
      checkpoint            the full path to a checkpoint file

    optional arguments:
      -h, --help            show this help message and exit
      --top_k TOP_K         top k results from classifier; integer number
      --category_names CATEGORY_NAMES
                            path to a *.json file
      --gpu                 pass this argument to use GPU
      --input_category INPUT_CATEGORY
                            the category of the input image
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', action="store", type=str, help='full path name to input image file')
    parser.add_argument('checkpoint', action="store", type=str, default='checkpoint.pth',help='the full path to a checkpoint file')
    parser.add_argument('--top_k', type=int, default=5,help='top k results from classifier; integer number')
    parser.add_argument('--category_names', type=str, default='',help='path to a *.json file')
    parser.add_argument('--gpu', action="store_true", default=False,help='pass this argument to use GPU')
    parser.add_argument('--input_category', type=str, default='', help='the category of the input image')
    return parser.parse_args()
#End of parse_input_args

def main():
    #Parse command line inputs
    in_arg = parse_input_args()

    # Load the model checkpoint
    print("\nLoading model checkpoint with name: {}\n".format(in_arg.checkpoint))
    model, accuracy, learnrate = checkpoint_load(in_arg.checkpoint)

    # Get the category names mapping if the file was provided
    if in_arg.category_names!='':
        with open(in_arg.category_names, 'r') as f:
            cat_to_name = json.load(f)
        model.class_to_idx = cat_to_name

    if in_arg.gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    # Predict the probabilities and top k classes
    print("Predict the image")
    probability_t, class_t = predict(in_arg.input, model, device, in_arg.top_k)
    
    probability = probability_t.tolist()[0]
    # If the category mapping file was provided, make a list of names
    if in_arg.category_names!='':
        classes = [model.class_to_idx[str(sorted(model.class_to_idx)[i])] for i in (class_t).tolist()[0]]
    # Otherwise create a list of index numbers
    else:
        classes = class_t.tolist()[0]
    
	# Print the predicted category to output
    print("\n=============================================================\n");
    print("Input image: {}".format(in_arg.input))
    print("Input category: {}".format(in_arg.input_category))
    print("Predicted category: {}".format(classes[0]))
    print("\n=============================================================\n");
    # Open the image
    image = process_image(Image.open(in_arg.input))

    # Plot the image and the top k probabilities in a horizontal bar chart
    fig, (ax1, ax2) = plt.subplots(figsize=(10,10), nrows=2)
    ax1 = imshow(image[0], ax1)
    ax1.axis('off')
    if in_arg.input_category!='':
        ax1.set_title("Input Image: {}".format(in_arg.input_category))
    else:
        ax1.set_title('Input Image:')
    ax2.barh(-np.arange(in_arg.top_k), probability, tick_label=classes)
    ax2.set_title("Predicted Class:\n(model accuracy: {:.3f})".format(accuracy))
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Category')
    plt.show()
#End of Main

if __name__ == '__main__':
    main()
