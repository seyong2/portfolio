---
title: Cat Face Detection
parent: MoodMeow - Cat Emotion Recognition
nav_order: 4
layout: default
---

## 1. The Goal: Finding the Cat's Face

Before extracting facial landmarks, I needed a reliable way to locate and crop a cat's face from an input photograph.

This step was more important than it might initially seem. A bounding box that was too small could cut off important facial features, such as the tips of the ears or the mouth. Since the next stages of my pipeline relied on these features, an inaccurate crop could affect everything that followed.

I decided to build an EfficientNetV2-based face detector, following the approach described in the research paper [*Automated Detection of Cat Facial Landmarks*](https://doi.org/10.1007/s11263-024-02006-w) by Martvel, Shimshoni, and Zamansky.

My experiments eventually led me to compare two loss functions for bounding-box regression: Smooth L1 loss and Complete Intersection over Union (CIoU) loss. In this article, I'll walk through the initial implementation, the limitations I discovered, and the results of my second experiment.

## 2. Why EfficientNetV2?

Rather than building a convolutional neural network from scratch, I decided to use a pretrained model.

My implementation used **EfficientNetV2-S with the default pretrained weights** provided by TorchVision:

`efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)`

EfficientNetV2 is a family of convolutional neural networks designed to improve training speed and model efficiency while maintaining strong image-recognition performance. The S variant is the smaller version of the EfficientNetV2 family, making it a practical starting point for a project with limited computational resources.

The pretrained weights are important because the model has already learned useful visual representations from ImageNet, a large image-classification dataset. Instead of learning every visual feature from scratch, I could start with a network that had already learned to recognize patterns such as edges, textures, shapes, and more complex visual structures.

Of course, recognizing ImageNet categories is different from locating a cat's face. The pretrained model could provide a useful feature extractor, but it still needed to learn the new task from cat-face images annotated with bounding boxes.

This is an example of **transfer learning**: reusing knowledge learned from one task as the starting point for another.

For reference, TorchVision documents the model and its pretrained weights here:

- [EfficientNetV2-S documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html)
- [EfficientNetV2 paper](https://arxiv.org/abs/2104.00298)

## 3. Adapting EfficientNetV2 for Bounding-Box Regression

The original EfficientNetV2-S model is designed for image classification. Its classification head produces scores for predefined categories. However, my task was different: given an image, I wanted the model to predict the location of a cat's face.

To achieve this, I needed to replace the original classification head with a regression head.

### How the modified architecture works

The research paper describes a model based on EfficientNetV2 in which the original top layers are removed and three fully connected layers are added, with sizes of 128, 64, and 4.

The architecture can be summarized as follows:

1. **Pretrained EfficientNetV2 feature extractor**: processes the input image and produces a representation of its visual features
2. **First fully connected layer (128 units)**: learns a transformation of the extracted features.
3. **Second fully connected layer (64 units)**: further transforms the representation before the final prediction.
4. **Output layer (4 units)**: predicts the coordinates of the bounding box.

ReLU activation functions are used in the hidden layers, while the final layer uses a linear activation so that it can output continuous coordinate values.

The four outputs represent the upper-left and lower-right corners of the bounding box:

$$[x_{min}, y_{min}, x_{max}, y_{max}]$$

These coordinates define a rectangle around the cat's face.





The first step in building a deep learning architecture for detecting cat facial landmarks, as outlined by Martvel, G. et al., was **identifying the cat’s face** in the input image. According to their paper, they rescaled images to 224 × 224 and fed them into a face detector. Their approach was based on an EfficientNetV2 model, which they modified by removing the top layers and adding three fully connected layers with ReLU and linear activation functions, sized 128, 64, and 4, respectively. Since the task was to predict the bounding box coordinates of the cat’s face, the final layer output four values representing the bounding box.

### Initial Experimentation

For initial experimentation, I followed a similar approach:

- **Resized the images to 224 $$\times$$ 224**, applying no additional transformations to speed up processing.
- **Split the dataset** into training (75%), validation (15%), and test (15%) sets.
- Used **SmoothL1Loss (also known as Huber loss)** as the loss function, which balances sensitivity to outliers while penalizing errors effectively-making it a better choice than Mean Squared Error (MSE) for this task.
- Optimized using the **Adam optimizer** with a learning rate of 0.0001.

Due to GPU's limitations, I trained the model for only two epochs. However, by using a **pre-trained model**, I was able to start with a low initial loss, which helped stabilize training. After training, the **validation loss dropped to 0.0013**, which I was quite pleased with.

### Evaluation

To assess the model's performance, I tested it on new cat images. The blue bounding box represents the ground truth, while the red box is the model's prediction. As seen below, the model performed well, accurately localizing the cats' faces.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9e9faf3f-d829-45d3-a9ca-5b574f9fbf33">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/0cd5f5cf-9ab3-4578-a5d6-eb44eac95d9f">
</p>

---
#### Resources
- Martvel, G., Shimshoni, I. & Zamansky, A. Automated Detection of Cat Facial Landmarks. Int J Comput Vis 132, 3103–3118 (2024). https://doi.org/10.1007/s11263-024-02006-w
- CatFLW Dataset. https://www.kaggle.com/datasets/georgemartvel/catflw/data
