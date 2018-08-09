# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

### Prerequisites
Image Classifier Project work done on Ubuntu 18.04 with anaconda package installed.

### Installing
To install dependencies in anaconda envirnment, please use below command: 
conda create --name <env> --file requirments.txt

## Running the tests
###1. How to use train.py
python train.py <Data Set Dir Absolute Path> --gpu (use --gpu for GPU) --epochs <Number> --arch <Architecture Name> --checkpoint <checkpoint.pth> 

Example:
python train.py /home/abhinema/Desktop/study/aipnd-project-master/flowers/ --gpu --epochs 1 --arch vgg19_bn --checkpoint checkpoint_vgg19_e1.pth 

###2. predict.py

python predict.py <Input Flower Name> <Classifier Name with Path> --gpu --category_names cat_to_name.json

Example:
python predict.py ./flowers/valid/1/image_06749.jpg ./image_classifier_model/checkpoint.pth --gpu --category_names cat_to_name.json

## Acknowledgments
 - Udacity's AIPND course ware for directly providing links and references to some of the apis.

 - 3Blue 1Brown youtube channel videos:
    https://www.watch.youtube/com?v=aircAruvnKk 

 - Conda commands: 
    https://conda.io/docs/

 - Various neural networks and basic introduction
    https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5   

 - for argparse:
    https://docs.python.org/3/library/argparse.html
    https://pymotw.com/3/argparse/
   
 - For passing boolean values as an argument 
     https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

 - Reference for Ploting images with matplot lib
     https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
     https://stackoverflow.com/questions/35286540/display-an-image-with-python/35286615
     https://matplotlib.org/users/pyplot_tutorial.html
     https://matplotlib.org/2.0.1/examples/pylab_examples/simple_plot.html
     
 - Last but not least, Credit goes to many anonymous authors/users for their fruitful input over internet. 
 
 ## Authors

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/abhinema/Image-Classifier/blob/master/LICENSE) file for details.

### Certificate
[embed]https://raw.githubusercontent.com/abhinema/Image-Classifier/blob/master/Udacity_Certificate.pdf[/embed]
