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

I found a paper by Martvel, Shimshoni, and Zamansky titled **"Automated Detection of Cat Facial Landmarks."** The paper was particularly relevant because described an approach for detecting the 48 facial landmarks in the CatFLW dataset.

Rather than using one model to predict all landmarks directly from an entire image, the authors broke the problem into several stages. They used separate EfficientNetV2-based models for different tasks, including face localization, facial region detection, and landmark detection.

The general idea was:

1. Detect and crop the cat's face.
2. Detect important facial regions.
3. Crop those regions.
4. Use separate models to detect landmarks within each region.
5. Combine the results into the final set of 48 landmarks.

This approach gave me a starting point for designing my own pipeline. Each model had a specific responsibility, allowing the overall problem to be broken down into smaller, more manageable tasks.

I adopted this general architecture for MoodMeow and added preliminary validation steps to ensure that the input image contained a cat and a suitable frontal face.

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

After confirming that the image contained a cat, I needed to determine whether it showed a suitable frontal face.

For this step, I used OpenCV's `haarcascade_frontalcatface` classifier. Its role was to check for the presence of a frontal cat face, rather than to perform the final face localization.

The Haar Cascade returns candidate bounding boxes for detected frontal cat faces and initially, I considered using this bounding box directly for the final face crop. However, I noticed that these bounding boxes sometimes excluded important parts of the cat's face, particularly the ears.

This was a problem for my project because the subsequent landmark detection models needed to analyze five facial regions: the left eye, right eye, left ear, right ear, and mouth. A crop that excluded part of the ears could prevent the corresponding model from detecting the necessary landmarks.

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

## Step 3: Precisely Localizing the Face

After checking for a frontal cat face, I needed to locate the face accurately enough for further analysis.

For this task, I used a dedicated **EfficientNetV2-based face localization model**, following the approach described in the research paper.

<p align="center">
  <img src="https://github.com/user-attachments/assets/916c2f79-ae9a-4f96-9938-d0d808c12226" title="face-detection">
</p>

The model predicts a bounding box around the cat's face. I then use this bounding box to crop the image, providing a more focused input for the subsequent stages.

This is different from using the bounding box produced by YOLOv7 or the Haar Cascade. YOLOv7 detects the cat rather than the face, while the Haar Cascade's bounding boxes sometimes exclude important facial features, such as the ears.

The face localization model is therefore responsible for obtaining the crop needed by the landmark detection pipeline.

## Step 4: Facial Region Detection

Once I had the face crop, the next task was to locate the individual facial regions.

Inspired by the research paper, I used a **separate EfficientNetV2-based model for facial region detection.**

This model identifies the regions needed for the next stage of landmark detection. In my implementation, these are:

* Left eye
* Right eye
* Left ear
* Right ear
* Mouth

Detecting these regions separately allows the following models to focus on the relevant part of the face instead of processing the entire facial image for every landmark prediction.

This region-based approach was particularly useful for my project because the final emotion inference depends on characteristics extracted from specific facial structures.

## Step 5: Landmark Detection with Region-Specific Models

After detecting the fire facial regions, I used a separate EfficientNetV2-based landmark detection model for each region.

Each model predicts the landmark coordinates associated with its assigned facial structure. For example, the left-eye model focused on the landmarks around the left eye, while the two ear models independently handle their respective ears.

The outputs from the five models are then combined to reconstruct the complete facial landmark representation.

The result is a set of **48 facial landmarks**, each represented by an (x) and (y) coordinate.

This multi-stage architecture follows the central idea of the research paper: divide a complex landmark detection problem into smaller tasks, then combine their outputs into a complete facial representation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d63e47e4-ed06-4a1b-8be5-0575f9f92e8c" title="ensemble-detection">
</p>

## What's Next?

At this point, I had a clear pipeline for transforming an input photograph into a structured set of 48 facial landmarks.

The research paper gave me a starting point for the face localization, facial region detection, and landmark detection stages. However, understanding the overall architecture was only the beginning. I still needed to implement and train the models, evaluate their performance, and make sure their predictions were accurate enough for the next stage of the project.

I decided to investigate each stage in more detail, starting with the model architecture and training process.

The next step was to look under the hood of the EfficientNetV2-based models: how they were designed, how they learned to predict bounding boxes or landmark coordinates, and how I evaluated their performance.

These experiments would determine whether I could reliably extract the facial information needed for the ultimate goal of MoodMeow: inferring a cat's emotional state from its facial characteristics.

---
#### Resources
- Martvel, G., Shimshoni, I. & Zamansky, A. Automated Detection of Cat Facial Landmarks. Int J Comput Vis 132, 3103–3118 (2024). https://doi.org/10.1007/s11263-024-02006-w
