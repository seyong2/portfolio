---
title: How to Detect Cat Facial Landmarks?
parent: Cat Emotion Recognition Journey
nav_order: 3
layout: default
---

In the previous post, we introduced five core emotions that cats can express. Now, before gathering cat image data for input, we need to outline how our emotion detection system will work. It’s not as simple as uploading an image and having the application instantly recognize the cat's emotion. We need to design an algorithm capable of performing this task. A key component of this is an automated process to detect the cat's face in an image and pinpoint specific facial features (ears, eyes, etc.) that are essential for classifying emotions. To achieve this, I found a study by Martvel, G. et al., which presents a deep learning architecture for detecting cat facial landmarks. I plan to reference this research and replicate their approach as a foundational element of my project.

To address the significant challenge researchers face in animal affective computing—namely, the lack of comprehensive, high-quality datasets—the authors of this paper introduced a dataset of cat facial images annotated with bounding boxes and 48 facial landmarks based on cat facial anatomy. Additionally, they implemented convolutional neural networks for detecting these landmarks, demonstrating excellent performance in the process. The steps involved in the landmark detection process are as follows.

### Face Detection

The initial step in landmark detection involves locating and cropping the cat's face from the input image. The authors utilized an EfficientNetV2 model, which takes the image as input and outputs a vector containing the four coordinates of the bounding box (representing the upper-left and lower-right corners).

<p align="center">
  <img src="https://github.com/user-attachments/assets/916c2f79-ae9a-4f96-9938-d0d808c12226" title="face-detection">
</p>

### Regions Detection

After detecting the face and cropping the image using the bounding box, the image is rescaled and processed to detect specific regions. A model similar to the face detector is then applied but with an output layer of size 10. This allows us to detect the coordinates of five key region centers: the eyes, nose (whiskers area), and both ears. To generate the training data, we averaged the landmark coordinates from these regions, using 5 landmarks out of the original 48.

### Ensemble Detection

Once the centers of the regions of interest are identified, the image is aligned by the eyes to reduce variations in roll tilt angles. Five fixed-size regions are then cropped from the image. Using fixed sizes ensures greater consistency in the data: if bounding boxes were automatically detected, the cropped areas would vary more in terms of position and relevant facial features. For instance, in this pipeline, the eye’s center is consistently located in the middle of the cropped eye region. These regions are resized to fit the input requirements of the EfficientNetV2 model.

The landmarks are then categorized by their respective regions: 8 landmarks for each eye, 5 for each ear, and 22 for the nose and whiskers area. Landmark detection is performed by an ensemble of five models, each resembling the main detector but with an output layer corresponding to twice the number of landmarks (to account for the $$x$$ and $$y$$ coordinates). The detected landmark coordinates are mapped back to the original image, forming a final vector of 96 coordinates (48 landmarks). The structure of the Ensemble Landmark Detector is shown below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d63e47e4-ed06-4a1b-8be5-0575f9f92e8c" title="ensemble-detection">
</p>

To wrap things up, I'll be working on replicating the architecture for detecting cat facial landmarks as outlined in the steps above. It’s definitely going to be a challenge, but I’m excited to dive in and I’m sure I’ll learn a lot along the way!

---
#### Resources
- Martvel, G., Shimshoni, I. & Zamansky, A. Automated Detection of Cat Facial Landmarks. Int J Comput Vis 132, 3103–3118 (2024). https://doi.org/10.1007/s11263-024-02006-w
