---
title: Cat Face Detector with CIoU Loss
parent: Cat Emotion Recognition Journey
nav_order: 4
layout: default
---

Previously, I built a deep learning model for detecting cat faces in input images, using a pre-trained EfficientNetV2 architecture. I fine-tuned the top layers for my task based on the research done by Martvel, G. et al. The model was optimized using the smooth L1 loss function, and after training, I made predictions on new images to evaluate how well the predicted bounding boxes enclosed the cat faces.

The results were acceptable in most images, but while preparing a new dataset for the next model (region detection), I noticed that in some cases, the predicted bounding boxes failed to capture key parts of the cat faces. This issue hindered my progress, prompting me to retrain the model and explore ways to improve its performance. I decided to experiment with a different loss function to see if it would yield better results. 

After researching, I found that Complete IoU (CIoU) loss is commonly used for bounding box regression problems in object detection tasks. If you're interested in learning more about this loss function and its comparison to others, check out my post on [Loss Functions for Object Detection](https://seyong2.github.io/portfolio/docs/data-science/computer-vision/iou-losses.html). I retrained the model with CIoU loss for 15 epochs and compared its performance with the previous model trained using smooth L1 loss. 

For comparison, I used Intersection over Union (IoU) on the test dataset to measure how well the predicted boxes overlapped with the ground truth. Although the model with smooth L1 loss had a reasonably good average IoU after just two epochs (since the validation loss was already quite low), I didn't retrain it further. The model trained with CIoU loss showed a clear improvement in IoU, producing bounding boxes that more accurately represented the target regions.

| with CIoU loss  | with Smooth L1 Loss |
| ------------- | ------------- |
| 0.8212  | 0.7521  |

To further illustrate the performance difference, let's look at some example images. The ground truth box is shown in blue, while the predicted bounding boxes from the CIoU model and the smooth L1 model are shown in red and green, respectively. As seen in the images, the green boxes (from the smooth L1 model) fail to fully capture important parts of the cat faces, which would create challenges later in the pipeline.

With the cat face detector now successfully set up, I can move on to the next step: region detection.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ac94002b-5403-42e2-a74a-e4a73b41d8a3" width="400">
  <img src="https://github.com/user-attachments/assets/064a93da-336a-4389-bf98-7c6b85090d66" width="400">
</p>
