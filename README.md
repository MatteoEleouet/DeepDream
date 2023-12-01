# DeepDream

## Overview
This project showcases the application of the DeepDream algorithm using different convolutional neural networks (CNNs). The DeepDream algorithm, originally developed by researchers at Google, manipulates the features learned by a neural network to create dream-like, surreal images. This project applies DeepDream to a single input image using three different models: VGG19, Vision Transformer (ViT), and InceptionV3. The transformations at each iteration are compiled into videos to visualize the evolution of the dream-like features.

Three gif are generated, showcasing the transformations with each iteration for the VGG19, ViT, and InceptionV3 models. These videos provide a fascinating look at how each model perceives and transforms the input image.
![DeepDream Prince GIF](deepdream_prince.gif)
![DeepDream Paris GIF](deepdream_paris02.gif)
![DeepDream Wallpaper GIF](deepdream_wallpaper.gif)

## DeepDream Algorithm
DeepDream is a technique which uses a convolutional neural network to find and enhance patterns in images, creating a dream-like appearance. The algorithm iteratively adjusts the input image to increase the activation of certain layers in the network.

Reference: Mordvintsev, A., Olah, C., & Tyka, M. (2015). Inceptionism: Going Deeper into Neural Networks. Google Research Blog. Retrieved from https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html

## Installation
To run the code, you need Python 3.x and the following dependencies:

- torch
- torchvision
- cv2 (OpenCV)
- numpy
- moviepy
- timm (for Vision Transformer)

You can install these packages using pip:
```bash pip install torch torchvision opencv-python numpy moviepy timm```

## Usage

1. Clone the repository.
2. Place the image you want to transform in the project directory.
3. Update img_path in the script with the path to your image.
4. Run the script:
```bash python deepdream_visualization.py```
5. The script will generate a video file deepdream_video.mp4 showing the transformation process for each model.

### Repository Structure
- **deepdream.py** : The main script with the DeepDream algorithm and video generation logic.
- **deepdream_video.mp4** : Video output of the transformations (generated after running the script).
- **your_image.jpg** : Example input image (replace with your image).

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details.

## Acknowledgments
This project is inspired by the work of Alexander Mordvintsev, Christopher Olah, and Mike Tyka on DeepDream. The implementation utilizes PyTorch, OpenCV, and other open-source software.

```bibtex
@article{mordvintsev2015inceptionism,
  title={Inceptionism: Going Deeper into Neural Networks},
  author={Mordvintsev, Alexander and Olah, Christopher and Tyka, Mike},
  year={2015},
  url={https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html}
}```
