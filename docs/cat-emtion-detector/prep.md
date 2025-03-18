---
title: How to Detect Cat Facial Landmarks?
parent: Cat Emotion Recognition Journey
nav_order: 3
layout: default
---

Now that we have a dataset ready to experiment with, we're moving on to the most exciting part-defining the framework for cat emotion detection. It's not as simple as providing an image and at instant you get a result like "*Your cat is feeling...*" (hope it's the case). Instead, we need to design a system capable of performing this task accurately. 

A crucial component of this system is an automated process to locate the cat's face in an image and identify its facial landmarks, which serve as the foundation for classifying emotions. To achieve this, I came across a paper from the same researchers who published the dataset I'll be using. In their work, they present a deep learning architecture for detecting cat facial landmarks-and it appears they used this architecture to annotate the dataset itself. 

One of the major challenges in animal affective computing is the lack of comprehensive, high-quality datasets. To address this, the authors introduced a dataset of cat facial images annotated with bounding boxes and 48 facial landmarks, cafefully selected based on cat facial anatomy. Additionally, they implemented convolutional neural networks (CNNs) for detecting these landmarks, achieving strong performance in the process. 

The landmark detection pipeline follows these steps:

### Face Detection

The first step in landmark detection involves **locating and cropping the cat's face** from the input image. The authors used an **EfficientNetV2 model**, which takes the image as input and outputs a bounding box defined by four coordinates (representing the upper-left and lower-right corners).

<p align="center">
  <img src="https://github.com/user-attachments/assets/916c2f79-ae9a-4f96-9938-d0d808c12226" title="face-detection">
</p>

### Regions Detection

Once the face is detected and cropped, the image is rescaled and processed to detect key facial regions. A model similar to the face detector is then applied, but with an **output layer of size 10** (accounting for both x and y coordinates), corresponding to the **coordinates of five key region centers**:

- Both eyes
- The nose (whiskers area)
- Both ears

To generate training data for this task, the authors **averaged the landmark coordinates from these regions**, selecting 5 representative points out of the original 48 landmarks.

### Ensemble Landmarks Detection

With the **centers of key regions** identified, the image is **aligned based on the eyes** to reduce variations in roll tilt angles. Then, five fixed-size regions are cropped, ensuring consistency in the detected features. This approach prevents unnecessary variations that could occur if bounding boxes were dynamically adjusted.

Each cropped region is resized to match the input requirements of the **EfficientNetV2 model**, and the **landmarks are categorized by region**:

- **8 landmarks per eye**
- **5 landmarks per ear**
- **22 landmarks for the nose and whiskers area**

Landmark detection is then performed by an **ensemble of five models**, each with an output layer corresponding to **twice the number of landmarks**. The detected landmarks are mapped back to the original image, forming a final output vector of **96 coordinates** (48 landmarks).

<p align="center">
  <img src="https://github.com/user-attachments/assets/d63e47e4-ed06-4a1b-8be5-0575f9f92e8c" title="ensemble-detection">
</p>

Now, it's time to put this into action! My next challenge is to replicate this architecture for detecting cat facial landmarks. It won’t be easy, but I’m excited to dive in—there’s a lot to learn, and I can’t wait to see where this takes me!

---
#### Resources
- Martvel, G., Shimshoni, I. & Zamansky, A. Automated Detection of Cat Facial Landmarks. Int J Comput Vis 132, 3103–3118 (2024). https://doi.org/10.1007/s11263-024-02006-w
