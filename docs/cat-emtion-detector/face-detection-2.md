---
title: Cat Face Detector with CIoU Loss
parent: Cat Emotion Recognition Journey
nav_order: 5
layout: default
---

In the previous post, I talked about **building a cat face detector** using a pre-trained EfficientNetV2. We saw that even after just two epochs, the model performed quite well on unseen cat images, achieving a low smooth L1 loss. 

However, as I tested the face detector on more images, I noticed that in some cases, the predicted **bounding boxes failed to capture key parts of cat's face**. Since images would later be cropped based on these bounding boxes, it was **crucial for the face detector to accurately identify the entire face**. To improve its performance, I explored different approaches and discovered a loss function that might be a better fit than Smooth L1 loss.

Complete Intersection over Union (CIoU) loss is commonly used for bounding box regression in object detection tasks. I won't go into detail about how CIoU works or why it might be a better choice thatn Smooth L1 loss here. If you're interested in learning more about loss functions for bounding box regression, check out my post on [Loss Functions for Object Detection](https://seyong2.github.io/portfolio/docs/data-science/computer-vision/iou-losses.html). 

### Comparing CIoU vs. Smooth L1 Loss

I retrained the face detection model using **CIoU loss** for 15 epochs and compared its performance against the previous model trained with **Smooth L1 loss**. To measure performance, I used **Intersection over Union (IoU)**, which evaluates how well the predicted bounding boxes overlap with the ground truth ones. 

Although the model trained with **Smooth L1 loss had a reasonably good average IoU after just two epochs**, I didn't continue training it further since the validation loss was already quite low. In contrast, the model trained with **CIoU loss** showed **clear improvements**, producing bounding boxes that more accurately represented the target regions.

| with CIoU loss  | with Smooth L1 Loss |
| ------------- | ------------- |
| 0.8212  | 0.7521  |

To visualize difference, let's look at some example images. The **ground truth box** is shown in **blue**, while the predicted bounding boxes from the **CIoU model** and the **Smooth L1 model** are shown in **red** and **green**, respectively. As seen in the images, the **green boxes (Smooth L1 model)** often **fail to fully capture** important parts of the cat's face, which would create issues in later processing steps.

With the **cat face detector** now successfully set up, I'll move on to the next step: **region detection**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ac94002b-5403-42e2-a74a-e4a73b41d8a3" width="400">
  <img src="https://github.com/user-attachments/assets/064a93da-336a-4389-bf98-7c6b85090d66" width="400">
</p>
