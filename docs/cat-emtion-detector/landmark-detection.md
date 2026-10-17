---
title: Facial Landmark Detection
parent: MoodMeow - Cat Emotion Recognition
nav_order: 6
layout: default
---

In the previous post, [Identifying 5 Cat Face Regions](https://seyong2.github.io/portfolio/docs/cat-emtion-detector/region-detection.html), I explained how I trained a model to predict the center points of five primary regions in a cat's face. These points provided approximate locations for the eyes, ears, and nose/mouth area.

The next step was to predict the detailed facial landmarks within each region. Rather than predicting all 48 landmarks directly from the entire face image, I followed a region-specific approach inspired by Martvel et al.'s research on automated cat facial landmark detection.

I trained five separate models, each responsible for predicting landmarks in one facial region. By combining their predictions, the pipeline could produce a complete set of 48 facial landmarks.

### 1. Cropping the Facial Regions

Before cropping the image into individual facial regions, I aligned the cat's face so that both eyes were approximately on the same horizontal level. This preprocessing step helped standardize the face orientation before extracting the smaller image patches.

Aligning the eyes provided a more consistent reference for the subsequent cropping and landmark-detection stages. After alignment, I used the predicted centers of the five facial regions to crop the corresponding image patches for the individual landmark models.

The region detector provided the predicted centers of the five facial regions. I used these centers to crop smaller image patches around the corresponding regions before passing them to the landmark detection models.

I used different crop sizes depending on the region:

| Facial region | Crop size | Number of landmarks | Output coordinates |
|---|---:|---:|---:|
| Left eye | 56 × 56 pixels | 8 | 16 |
| Right eye | 56 × 56 pixels | 8 | 16 |
| Left ear | 112 × 112 pixels | 5 | 10 |
| Right ear | 112 × 112 pixels | 5 | 10 |
| Nose and mouth area | 112 × 112 pixels | 22 | 44 |
| **Total** | — | **48** | **96** |

The eye regions used smaller crops, while the ears and nose/mouth area used larger ones. This allowed each model to focus on its assigned facial region instead of processing the entire face.

The predicted region centers were not themselves the final landmarks. They served as reference points for extracting the image patches in which the detailed landmarks would be detected.

Because the landmarks coordinates had to correspond to these cropped images during training, I also adjusted the landmark coordinates to match each crop. This was an essential part of preparing the training targets: the image patch and its target coordinates needed to use the same coordinate system.

### 2. Training Five Landmark Detection Models

For landmark detection, I used the same general training setup as in the region-detection stage, with a separate model for each facial region.

I used the EfficientNetV2 architecture and modified its output layer to predict the required number of coordinate values. Unlike the region detector, which predicted ten values representing the centers of five regions, each landmark detector predicted the detailed coordinates for only its assigned region.

Each landmark is represented by an $$x$$-coordinate and a $$y$$-coordinate. Consequently, a model predicting eight landmarks produces 16 coordinate values, while a model predicting 22 landmarks produces 44.

The five models therefore had different output dimensions:

* Left eye detector: 16 coordinate values for 8 landmarks.
* Right eye detector: 16 coordinate values for 8 landmarks.
* Left ear detector: 10 coordinate values for 5 landmarks.
* Right ear detector: 10 coordinate values for 5 landmarks.
* Nose and mouth detector: 44 coordinate values for 22 landmarks.

This design allowed each model to specialize in a smaller facial region. The eye detectors, for example, only needed to predict the points associated with their respective eyes, rather than learning to locate all 48 points simultaneously.

#### Training configuration

I reused the training configuration from the region detector. The main difference was the number of outputs required for each model.

The models were trained to predict continuous coordinate values, making landmark detection a regression task rather than a classification task. The training targets were the annotated landmark coordinates adjusted to correspond to the respective image crops.

### 3. Combining the Predictions

Once the five models generated their predictions, their outputs represented the detailed landmarks for their respective regions.

The final set contained 48 landmarks:

* 8 points for the left eye
* 8 points for the right eye
* 5 points for the left ear
* 5 points for the right ear
* 22 points for the nose and mouth area

Together, these points described the geometry of the cat's face in greater detail than the five region centers alone.

An important distinction is that the landmark models operated on cropped image patches. I did not need to convert their predictions back into the original full-face coordinate system for the next stage of my implementation. Instead, I used the predicted coordinates to calculate geometric features for emotion inference.

This kept the landmark-detection stage focused on identifying facial points, while the following stage was responsible for extracting meaningful measurements from those points.

### 4. Evaluating the Predictions

I inspected the models' predictions on test images by comparing the predicted landmarks with the ground-truth annotations. In the visualizations below, the red points represent the model predictions.

<p align="center">
  <img src="https://github.com/user-attachments/assets/cc8052bf-13ab-4a68-9fe3-4d9f895b3613" width=200>
  <img src="https://github.com/user-attachments/assets/bbf20aaa-f376-4e58-a622-e5b45b671462" width=200>
  <img src="https://github.com/user-attachments/assets/207f3071-4881-424e-9655-c05e2498362e" width=200>
  <img src="https://github.com/user-attachments/assets/add2ee33-6cec-4c30-bab7-35250c2ade0b" width=200>
  <img src="https://github.com/user-attachments/assets/55adf724-8b44-4e93-8f01-f1e5df2045a3" width=200>
</p>

In the displayed examples, the predicted points appear close to the corresponding annotated landmarks. These visual checks helped me assess whether the models were locating the expected facial features.

However, the visualizations represent only a small number of examples. Without a recorded quantitative evaluation across the test set, I cannot conclude from these images alone that the models perform consistently or reliably on unseen cat images.

The examples nevertheless illustrate the intended outcome of the approach: using five region-specific models to predict detailed facial landmarks from localized image patches.

### 5. From Facial Landmarks to Emotion Features

With the 48 facial landmarks predicted, the landmark-detection pipeline was complete. The next step was to use their coordinates to extract geometric features associated with the cat's facial appearance.

For example, the relative positions of points around the eyes can help describe eye shape and openness, while the positions of the ear landmarks can provide information about ear orientation. The points around the nose and mouth provide additional information about the central facial region.

These measurements formed an intermediate representation between the original cat image and MoodMeow's emotion-inference logic. Rather than relying only on the raw image, the following stage could use geometric information derived from the predicted landmarks.

It is important to distinguish these measurable facial characteristics from emotion itself. Facial landmarks describe the positions of visible features; they do not directly establish what a cat is feeling. MoodMeow uses the extracted characteristics as part of an entertainment-oriented process for inferring a possible mood.

In the next article, I will bring the pipeline together and explain how the facial information is used to infer a cat's mood.

---
#### Resources
- Martvel, G., Shimshoni, I. & Zamansky, A. Automated Detection of Cat Facial Landmarks. Int J Comput Vis 132, 3103–3118 (2024). https://doi.org/10.1007/s11263-024-02006-w
