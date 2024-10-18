---
title: Loss Functions for Object Detection
parent: Computer Vision
nav_order: 2
layout: default
---

This post covers various loss functions used in object detection tasks. In general, object detection requires two types of loss functions: one for **object classification** (e.g., determining whether an object is a cat or a dog) and another for **bounding box regression**. We'll focus primarily on the latter, which measures how accurately the predicted bounding boxes enclose objects.

## Traditional Regression Loss Functions: MSE and MAE

When it comes to regression problems, **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** might be the first loss functions that come to our mind- and they can be used for the bounding box regression as well. A bounding box is defined by four coordinates: $$(x_1, y_1, x_2, y_2)$$, where $$(x_1, y_1)$$ represents the top-left corner and $$(x_2, y_2)$$ represents the bottom-right corner. Given two bounding boxes- the **ground truth** box (blue) and the **predicted box** (red)- the goal is to calculate how close the predicted bounding box is to the ground truth by comparing the coordinates of the two boxes.

<p align="center">
  <img src="https://github.com/user-attachments/assets/21d9f801-8ddc-44bb-8bd5-ca541fbf63aa">
</p>

However, MSE and MAE have limitations. One major issue is that they are **not scale-invariant**. When an image is scaled up or down, the ground truth and predicted bounding boxes change accordingly, but the loss values also vary based on the scale. Specifically, as the image is scaled down, both the MSE and MAE losses decrease, even if the relative positions of the boxes remain the same. This suggests a need for loss functions that are robust to changes in scale.

Another limitation is that bounding boxes optimized using MSE or MAE do not always have high quality. Consider the figure below: we have two examples, each with three predicted bounding boxes. For each predicted bo, the losses are calculated using MSE (denoted as $$\|\|.\|\|_2$$) and MAE (denoted as $$\|\|.\|\|_1$$), alongside alternative loss functions such as IoU and GIoU (discussed later). In example (a), all three black bounding boxes have the same MSE loss, but clearly, the first box has worse quality compared to the other two.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fa14d63d-6c86-44a2-aa60-613d2d76edeb">
</p>

### Intersection Over Union (IoU)

**Intersection over union (IoU)**, also known as the Jaccard Index, is a widely used metric to measure the similarity between sets- in this case, bounding boxes. As the name implies, the IoU is calculated as the area of overlap between the predicted and ground truth boxes divided by the area of their union. IoU values range from 0 and 1, where:

- 0 means the boxes do not overlap at all.
- 1 means the boxes overlap perfectly.
  
<p align="center">
  <img src="https://github.com/user-attachments/assets/fb4f8225-72f1-4f5c-b27b-e81af0b3c80f">
</p>

To convert IoU into a loss function, we simply subtract it from 1. In this way, if the bounding boxes overlap perfectly, the loss will be 0; if they do not overlap, the loss will be 1.

$$L_{IoU}=1-IoU$$

However, IoU loss has a significant drawback: when there is **no overlap** between the ground truth and predicted boxes, the IoU is always zero, regardless of how far apart the boxes are. In such cases, the loss remains constant at 1, and since the gradient is zero, the model struggles to converge. This motivates the need for alternative loss functions that address this issue, such as **Generalized IoU (GIoU)**, which will be discussed next.

### Generalized IoU (GIoU)


#### Resources
- H. Rezatofighi et al., "Generalized Intersection Over Union: A Metric and a Loss for Bounding Box Regression," in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, 2019, pp. 658-666, doi: 10.1109/CVPR.2019.00075.
