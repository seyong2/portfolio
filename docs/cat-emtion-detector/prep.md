---
title: From Cat Detection to Facial Landmarks
parent: MoodMeow - Cat Emotion Recognition
nav_order: 3
layout: default
---

Now that I had a dataset and a framework for understanding feline emotional states, I needed to solve a more practical problem:

**How do I turn a random cat photo into the facial information needed to infer an emotional state?**

It is not enough to simply give an image to a model and expect an answer like *"Your cat is feeling happy."*

Before I can analyze the cat's eyes, ears, and mouth, I first need to find the cat, locate its face, and identify the relevant facial features.

This led me to build a multi-stage computer vision pipeline.

```text
Cat image
    ↓
Cat Detection
    ↓
Frontal Face Check
    ↓
Face Detection
    ↓
Face Crop
    ↓
Five Facial Regions
    ↓
Region-specific Landmark Detection
    ↓
48 Facial Landmarks
    ↓
Feature Extraction
    ↓
Emotion Inference
```

## Learning from Existing Research

To understand how cat facial landmarks could be detected automatically, I looked for research specifically focused on feline facial landmark detection.

I found a paper by Martvel, Shimshoni, and Zamansky titled **"Automated Detection of Cat Facial Landmarks."** The paper was particularly relevant because it came from the same research group behind the CatFLW dataset I had just selected.

The authors proposed a deep learning pipeline for detecting the 48 facial landmarks in CatFLW. Their approach breaks the problem into several stages rather than trying to predict all 48 landmarks directly from the entire image.

The general idea was:

1. Detect and crop the cat's face.
2. Detect important facial regions.
3. Crop those regions.
4. Use separate models to detect landmarks within each region.
5. Combine the results into the final set of 48 landmarks.

This gave me a strong starting point for my own landmark detection system.

However, there was still an important problem to solve before I could even get to the face.

## Step 1: Detecting the Cat

The input to MoodMeow is simply a photograph uploaded by the user. Since the user can provide an arbitrary image, I first needed to make sure that the image actually contained a cat.

For this step, I used **YOLOv7**, a real-time object detection model. YOLOv7 can identify objects in an image and return bounding boxes around them, so technically I could have used its predicted cat bounding box to crop the image.

However, I decided not to use the cat bounding box as the input for the facial analysis stage.

There were two main reasons.

### A cat bounding box is not necessarily a useful crop

First, detecting a cat does not mean that the cat is showing its face.

For example, a photograph might show a cat from behind or from the side. YOLOv7 could correctly detect the cat and produce an accurate bounding box, but cropping around the box would still give me an image that is not useful for facial landmark detection.

I did not use YOLOv7 to perform the facial analysis or to obtain the face crop used by landmark models.

Instead it acts as the first gate in the pipeline:

```text
Cat detected ≠ Usable cat face
```

### The final objective is a face crop, not a cat crop

Second, even when the cat is facing the camera, the YOLOv7 bounding box represents the entire cat, not just its face.

The downstream land mark models do not need the cat's body. They need a relatively precise crop of the facial area.

Using the cat bounding box would therefore add another unnecessary step:

```text
Cat image
    ↓
Cat bounding box
    ↓
Large cat crop
    ↓
Find the face
    ↓
Face crop
```

Instead, I wanted the pipeline to move directly from cat detection to face validation and localization.

Therefore, I used YOLOv7 primarily as an initial validation step:

```text
Input image
    ↓
YOLOv7
    ↓
Does the image contain a cat?
    ↓
Yes → continue
No → stop
```

Once a cat was detected, I could focus on the more specific question: **does the image contain a usable frontal cat face?**

## Step 2: Checking for a Frontal Cat Face

For this step, I used OpenCV's `haarcascade_frontalcatface` classifier.

The purpose of the Haar Cascade in my pipeline was different from that of YOLOv7. Rather than simply checking whether there was a cat somewhere in the image, it allowed me to check whether a **frontal cat face** was present.

The Haar Cascade returns candidate bounding boxes for detected frontal cat faces. I initially considered using this bounding box directly for the final face crop, but its localization was not accurate enough for the precision required by the landmark detection models.

Therefore, I used the Haar Cascade primarily as a **frontal-face validation step**, while a separate face detector was responsible for obtaining the precise face bounding box.

The resulting logic was:

```text
Input image
     ↓
YOLOv7
     ↓
Cat detected?
     ↓ Yes
Haar Cascade
     ↓
Frontal face detected?
     ↓ Yes
Face detector
     ↓
Precise face crop

```

The distinction is important because the goal of this stage is not simply to find a cat. **The goal is to obtain a clean, usable crop of the cat's face.**

This is a good example of something I learned while building the project: a model does not necessarily need to solve the entire problem to be useful. Sometimes a relatively simple model can work well as one component of a larger pipeline.

------
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

Once the face is detected and cropped, the image is rescaled and processed to detect key facial regions. A model similar to the face detector is then applied, but with an **output layer of size 10** (accounting for both $$x$$ and $$y$$ coordinates), corresponding to the **coordinates of five key region centers**:

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
