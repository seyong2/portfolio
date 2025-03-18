---
title: Detecting Cat's Face
parent: Cat Emotion Recognition Journey
nav_order: 4
layout: default
---

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
