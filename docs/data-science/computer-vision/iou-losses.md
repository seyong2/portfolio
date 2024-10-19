---
title: Loss Functions for Object Detection
parent: Computer Vision
nav_order: 2
layout: default
---

This post covers various loss functions used in object detection tasks. In general, object detection requires two types of loss functions: one for **object classification** (e.g., determining whether an object is a cat or a dog) and another for **bounding box regression**. We'll focus primarily on the latter, which measures how accurately the predicted bounding boxes enclose objects.

### Traditional Regression Loss Functions: MSE and MAE

When it comes to regression problems, **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** might be the first loss functions that come to our mind- and they can be used for the bounding box regression as well. A bounding box is defined by four coordinates: $$(x_1, y_1, x_2, y_2)$$, where $$(x_1, y_1)$$ represents the top-left corner and $$(x_2, y_2)$$ represents the bottom-right corner. Given two bounding boxes- the **ground truth** box (blue) and the **predicted box** (red)- the goal is to calculate how close the predicted bounding box is to the ground truth by comparing the coordinates of the two boxes.

<p align="center">
  <img src="https://github.com/user-attachments/assets/21d9f801-8ddc-44bb-8bd5-ca541fbf63aa">
</p>

However, MSE and MAE have limitations. One major issue is that they are **not scale-invariant**. When an image is scaled up or down, the ground truth and predicted bounding boxes change accordingly, but the loss values also vary based on the scale. Specifically, as the image is scaled down, both the MSE and MAE losses decrease, even if the relative positions of the boxes remain the same. This suggests a need for loss functions that are robust to changes in scale.

Another limitation is that bounding boxes optimized using MSE or MAE do not always have high quality. Consider the figure below: we have two examples, each with three predicted bounding boxes. For each predicted bo, the losses are calculated using MSE (denoted as $$\|.\|_2$$) and MAE (denoted as $$\|.\|_1$$), alongside alternative loss functions such as IoU and GIoU (discussed later). In example (a), all three black bounding boxes have the same MSE loss, but clearly, the first box has worse quality compared to the other two.

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

To address the limitation of IoU loss, **Generalized IoU (GIoU)** introduces an additional term that considers the distance between the non-overlapping bounding boxes. GIoU is defined as follows:

$$ GIoU = IoU - \frac{C-(AUB)}{C}$$

where $$A$$ and $$B$$ represent the two bounding boxes, and $$C$$ is the smallest box that encloses both $$A$$ and $$B$$. Even when the ground truth and predicted bounding boxes don't overlap, GIoU does not remain at 0. Instead, it produces a negative value, with the magnitude depending on the distance between the boxes. The further apart they are, the smaller the GIoU value becomes (but never smaller than -1, since the term $$\frac{C-(AUB)}{C}$$ is always less than 1).

<p align="center">
  <img src="https://github.com/user-attachments/assets/5d66af80-73b4-4f99-a903-aa42e7b8e696">
</p>

We can convert GIoU into a loss function by subtracting it from 1:

$$L_{GIoU}=1-GIoU=1-(IoU - \frac{C-(AUB)}{C})$$

Because GIoU can take on non-zero values even when the boxes do not overlap, the corresponding loss will still have a gradient, allowing the model to adjust the bounding box in such cases.

### Distance IoU (DIoU)

While GIoU mitigates the gradient vanishing problem for non-overlapping boxes, it has other drawbacks. For example, when the predicted box does not overlap with the target, GIoU first tries to increase the size of the predicted box to create overlap, reducing in $$\frac{C-(AUB)}{C}$$. Only then does it try to improve the IoU by maximizing the overlap area, which can be inefficient and lead to inaccurate predictions. In the figure below, we observe that three predicted boxes have the same IoU and GIoU losses, even though the last prediction is the most accurate in terms of center alignment.

<p align="center">
  <img src="https://github.com/user-attachments/assets/924232a3-602d-4d8d-9def-cf6eb5251e28">
</p>

Additionally, when the two bounding boxes are horizontally or vertically aligned with no overlap, GIoU degrades to IoU, causing the same gradient vanishing problem.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3fd191a8-32be-46d2-aa2e-0fbe5b18a509">
</p>

To address these issues,  **Distance IoU (DIoU)** introduces a penalty term that minimizes the normalized distance between the centers of two bounding boxes. Then, the DIoU loss is defined as:

$$ L_{DIoU}=1-(IoU-\frac{\rho^2(b, b^{gt})}{c^2})= 1-IoU+\frac{\rho^2(b, b^{gt})}{c^2}$$ 

where $$b$$ and $$b^{gt}$$ denote the center points of the predicted and target bounding boxes, $$\rho(\cdot)$$ is the Euclidean distance between these points, and $$c$$ is the diagonal length of the smallest enclosing box covering both boxes. Dividing the penalty term by $$c$$ makes the DIoU scale invariant. The penalty term is scale-invariant, helping DIoU to converge faster than GIoU by directly minimizing the center point distance. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/18599971-c09a-46b5-ac5e-ee866af49a2b">
</p>

### Complete IoU (CIoU)

The final loss function that we'll explore is **Complete IoU (CIoU)** loss, which extends DIoU loss by accounting for the aspect ratio of the bounding boxes. Aspect ratio alignment is important in many cases, so CIoU adds this consideration to the loss function:

$$ L_{CIoU}= 1-IoU+\frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

Here, $$\alpha$$ is a trade-off parameter, and $$v$$ measures the consistency of aspect ratios between the predicted and ground truth boxes.

$$v=\frac{4}{\pi^2}(arctan\frac{w^{gt}}{h^{gt}}-arctan\frac{w}{h})^2$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/03fab6aa-745e-4141-a6d6-8ae92a153e51">
</p>

The parameter $$\alpha$$ is defined as:

$$\alpha=\frac{v}{(1-IoU)+v}$$

When $$\alpha$$ is large, meaning that the boxes overlap perfectly, the loss emphasizes aspect ratio alignment, penalizing mismatched shapes more heavily. When $$\alpha$$ is small or zero, the loss focuses more on IoU and center point distance, ignoring shape mismatches.

### Comparison between Different IoU Loss Functions

Finally, let's compare the performance of the different IoU loss functions we've discussed. As shown in the figure below, IoU performs poorly even after 200 iterations. GIoU shows better performance but lags behind DIoU and CIoU, both of which converge faster and result in more accurate bounding boxes.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3c8e60fa-f72e-4fd1-ad93-5698daa53fba">
</p>

#### Resources
- H. Rezatofighi et al., "Generalized Intersection Over Union: A Metric and a Loss for Bounding Box Regression," in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, 2019, pp. 658-666, doi: 10.1109/CVPR.2019.00075.
- Zheng, Zhaohui, et al. "Distance-IoU loss: Faster and better learning for bounding box regression." Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 07. 2020.
