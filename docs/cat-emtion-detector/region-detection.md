---
title: Identifying 5 Cat Face Regions
parent: Cat Emotion Recognition Journey
nav_order: 6
layout: default
---

In [How to Detect Cat Facial Landmarks?](https://seyong2.github.io/portfolio/docs/cat-emtion-detector/prep.html), the second step in the **cat emotion detection framework** is to **identify the center points of five primary facial regions:

- Two eyes
- Nose and mouth
- Two ears
-

### Computing Region Centers

To build a **cat region detection model**, we first need to **compute the center points for each region. This is done by **averaging the individual landmark coordinates** within each region. 

To illustrate, consider this cat image:

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ceb8173-64aa-45cf-b3e5-1f58ecfb3104" width="400">
</p>

- Each **ear** contains 5 landmarks
- Each **eye** contains 8 landmarks
- The **nose area** contains 22 landmarks

By calculating the **average coordinates** of these landmarks, we determine a **single center point per region**, which will be used to **train the region detector**. 

### Training the Region Detector

To **increase model robustness**, we **randomly rotate** training images. This augmentation helps the detector perform well even when cat faces appear at **different angles**.

As with the **face detector**, we use a **pre-trained EfficientNetV2 model**, optimizing it with the **Mean Squared Error (MSE) loss function**, since this is a regression task.

### Performance Evaluation

To assess the model’s accuracy, let's examine two test images. The **blue points** represent the **ground truth centers**, while the **red points** are the **model’s predictions**. As seen in the images, the predicted points are very close to the ground truth, indicating that the model performs well.

<p align="center">
  <img src="https://github.com/user-attachments/assets/78718d24-6ea2-4a85-9b45-ac2e43875189" width="400">
  <img src="https://github.com/user-attachments/assets/6176c360-965e-4f22-b62d-9ea208d9dfa0" width="400">
</p>

With the region detector successfully trained, the next step is to **refine landmark detection for emotion classification**-brining us closer to our goal of **analyzing cat emotions**.
