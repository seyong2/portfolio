---
title: Detecting Cat's Face
parent: Cat Emotion Recognition Journey
nav_order: 4
layout: default
---

The first step taken by Martvel, G. et al. in building a deep learning architecture for detecting cat facial landmarks was to identify the location of the cat's face in the input image. According to their paper, they rescaled the input images to a resolution of 224 $$\times$$ 224 and fed them into the face detector. The team used an EfficientNetV2 model for face detection, modifying the architecture by removing the top layers and adding three fully connected layers with ReLU and linear activation functions, sized 128, 64, and 4, respectively. Since the face detector's task is to predict the bounding box coordinates for the cat's face, the final layer outputs four values.

Following this framework, I attempted to replicate their work in Pytorch using the same dataset that the authors created: the CatFLW dataset. It consists of 2,079 images of cats' faces captured in various environments and conditions, annotated with 48 facial landmarks and a bounding box on the cat’s face. The dataset can be easily downloaded [here](https://www.kaggle.com/datasets/georgemartvel/catflw/data).

For initial experimentation, I resized the images to 224 $$\times$$ 224 and applied no additional transformations to speed up the process. The dataset was split into training (75%), validation (15%), and test (15%) sets. I used SmoothL1Loss (also known as Huber loss) as the loss function, which is ideal for regression tasks where you want to be less sensitive to outliers than Mean Squared Error (MSE) but still penalize errors. For optimization, I employed the Adam optimizer with a learning rate of 0.0001.

Due to my GPU's limitations, I trained the model for only two epochs. However, starting with a pre-trained model allowed me to begin with a low initial loss. After training, the validation loss dropped to 0.0013, which I was quite pleased with. To evaluate the model, I tested it on new cat images. The blue bounding box represents the ground truth, while the red box is the model's prediction. As seen below, the model performed well, accurately localizing the cats' faces.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9e9faf3f-d829-45d3-a9ca-5b574f9fbf33">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/0cd5f5cf-9ab3-4578-a5d6-eb44eac95d9f">
</p>

#### Resources
- Martvel, G., Shimshoni, I. & Zamansky, A. Automated Detection of Cat Facial Landmarks. Int J Comput Vis 132, 3103–3118 (2024). https://doi.org/10.1007/s11263-024-02006-w
- CatFLW Dataset. https://www.kaggle.com/datasets/georgemartvel/catflw/data
