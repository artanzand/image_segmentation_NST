# Semantic Image Segmentation with U-Net

Author: Artan Zandian  
Date: February 2022

## About

This project is a second phase of my [Neural style Transfer project](https://github.com/artanzand/neural_style_transfer) where I apply semantic image segmentation on the content image to filter out the images of people from the stylized image. Please refer to Neural Style Transfer [post](https://artanzand.github.io//neural-style-transfer/) for more information on implementation of that project.
<br>

This project includes a Tensorflow implementation of the U-Net Architecture used for image segmentation which was originally introduced by Ronneberger et al. in a [paper](https://arxiv.org/abs/1505.04597) in 2015. For review of the framework and to understand how each individual piece works please refer to [my project post](https://artanzand.github.io//sematic-segmentation/). I have also included a [notebook file](https://github.com/artanzand/image_segmentation_NST/blob/main/src/U-Net_model.ipynb) which will walk you through the image loading, data wrangling, building of U-Net architecture and training of the model.
<p align="center">
  <img src="https://github.com/artanzand/image_segmentation_NST/blob/main/examples/evolution.gif" />
</p>

## Dataset

Person Segmentation [dataset](https://www.kaggle.com/nikhilroxtomar/person-segmentation)
<br>

## Usage

### Cloning the Repo

Clone this Github repository and install the dependencies by running the following commands at the command line/terminal from the root directory of the project:

```conda env create --file environment.yaml```  
```conda activate NST```

Run the below command by replacing content image , style image and save directory.  
```python stylize.py --content <content image> --style <style image> --save <save directory>```

Run `python stylize.py --help` to see a list of all options and input type details.  
  
Two optional arguments of --similarity (default "balanced") and --epochs (default 500) control the similarity to either of input photos and number of iternations respectively.
For a 512??680 pixel content file, 1000 iterations take 75 seconds on an Nvidia Jetson Nano 2GB, or 75 minutes on an Intel Core i5-8250U. Due to the speedup using a GPU is highly recommended.

### Downloading and Training the Model

To use the Kaggle API, sign up for a Kaggle account at <https://www.kaggle.com>. Then go to the 'Account' tab of your user profile (<https://www.kaggle.com/><username>/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location `~/.kaggle/kaggle.json`. I have already included the `kaggle` package in the repo environment, and running the below script should download the required files.

You should now move to `/src` directory to download the data, train and save the U_Net model. You will need 1GB of memory for the data, and 100 MB of memory to store the model. It is optional to keep the data once the training is done. Run the following commands at the command line/terminal from the `/src` directory of the project to download the data files and augment them in a `/data` folder, and finally train the model. This will give you a `Unet_model.h5` in the `model/` directory. The training should take about 45 minutes on an NVIDIA Tesla P100 on the cloud.

```
python download_data.py --dataset=nikhilroxtomar/person-segmentation --file_path=../data/
python train.py
rm -r ../data
```

### Weights File

An alternative to using the script above to download and train the model would be to download the trained model and weights from [here](https://drive.google.com/drive/u/0/my-drive). Make sure the model is saved in a `/model` folder under the root directory.

## Examples of Image Segmentation

The `predict()` function can be used on its own for image segmentation.

```
python src/predict.py --file_path=../examples/alberta_walking.jpg
```

<p align="center">
  <img src="https://github.com/artanzand/image_segmentation_NST/blob/main/examples/predict_output.JPG" />
</p>

Below we see a comparison of the real training mask versus the one predicted by the model.
<p align="center">
  <img src="https://github.com/artanzand/image_segmentation_NST/blob/main/examples/true_v_predicted.JPG" />
</p>
<br>

## Requirements

### Neural Network Model and Weights

The trained U-Net model and weights need to be downloaded from [here](https://drive.google.com/drive/u/0/my-drive) and saved in a `/model` folder under the root directory. Alternatively, the scripts in the Usage section can be used to download and train the model.

The main function in `stylize.py` loads [VGG19 Architecture](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19/VGG19) from Keras with ImageNet weights.  

### Dependencies  

A complete list of dependencies is available
[here](https://github.com/artanzand/neural_style_transfer/blob/main/environment.yaml).
<br>- python=3.9
<br>- tensorflow=2.6
<br>- keras=2.6
<br>- docopt=0.6.1
<br>- pillow=8.4.0
<br>- numpy=1.22.0
<br>- imageio=2.6.1

## License

This project is licensed under the terms of the MIT license.

## Credits and References

[1] Gatys, Leon A., Ecker, Alexander S. and Bethge, Matthias. "A Neural Algorithm of Artistic Style.." CoRR abs/1508.06576 (2015): [link to paper](https://arxiv.org/abs/1508.06576)  
[2] Image Segmentation with DeepLabV3Plus: [link to repo](https://github.com/nikhilroxtomar/Human-Image-Segmentation-with-DeepLabV3Plus-in-TensorFlow)  

[3] [DeepLearning.ai](https://www.deeplearning.ai/) Deep Learning Specialization lecture notes  

[4] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.: [link to paper](https://arxiv.org/abs/1505.04597)
