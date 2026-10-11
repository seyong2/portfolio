---
title: Cat Face Detection
parent: MoodMeow - Cat Emotion Recognition
nav_order: 4
layout: default
---

## 1. The Goal: Finding the Cat's Face

Before extracting facial landmarks, I needed a reliable way to locate and crop a cat's face from an input photograph.

This step was more important than it might initially seem. A bounding box that was too small could cut off important facial features, such as the tips of the ears or the mouth. Since the next stages of my pipeline relied on these features, an inaccurate crop could affect everything that followed.

I decided to build an EfficientNetV2-based face detector, following the approach described in the research paper [*Automated Detection of Cat Facial Landmarks*](https://doi.org/10.1007/s11263-024-02006-w) by Martvel, Shimshoni, and Zamansky.

My experiments eventually led me to compare two loss functions for bounding-box regression: Smooth L1 loss and Complete Intersection over Union (CIoU) loss. In this article, I'll walk through the initial implementation, the limitations I discovered, and the results of my second experiment.

## 2. Why EfficientNetV2?

Rather than building a convolutional neural network from scratch, I decided to use a pretrained model.

My implementation used **EfficientNetV2-S with the default pretrained weights** provided by TorchVision:

`efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)`

EfficientNetV2 is a family of convolutional neural networks designed to improve training speed and model efficiency while maintaining strong image-recognition performance. The S variant is the smaller version of the EfficientNetV2 family, making it a practical starting point for a project with limited computational resources.

The pretrained weights are important because the model has already learned useful visual representations from ImageNet, a large image-classification dataset. Instead of learning every visual feature from scratch, I could start with a network that had already learned to recognize patterns such as edges, textures, shapes, and more complex visual structures.

Of course, recognizing ImageNet categories is different from locating a cat's face. The pretrained model could provide a useful feature extractor, but it still needed to learn the new task from cat-face images annotated with bounding boxes.

This is an example of **transfer learning**: reusing knowledge learned from one task as the starting point for another.

For reference, TorchVision documents the model and its pretrained weights here:

- [EfficientNetV2-S documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html)
- [EfficientNetV2 paper](https://arxiv.org/abs/2104.00298)

## 3. Adapting EfficientNetV2 for Bounding-Box Regression

The original EfficientNetV2-S model is designed for image classification. Its classification head produces scores for predefined categories. However, my task was different: given an image, I wanted the model to predict the location of a cat's face.

To achieve this, I needed to replace the original classification head with a regression head.

### How the modified architecture works

The research paper describes a model based on EfficientNetV2 in which the original top layers are removed and three fully connected layers are added, with sizes of 128, 64, and 4.

The architecture can be summarized as follows:

1. **Pretrained EfficientNetV2 feature extractor**: processes the input image and produces a representation of its visual features
2. **First fully connected layer (128 units)**: learns a transformation of the extracted features.
3. **Second fully connected layer (64 units)**: further transforms the representation before the final prediction.
4. **Output layer (4 units)**: predicts the coordinates of the bounding box.

ReLU activation functions are used in the hidden layers, while the final layer uses a linear activation so that it can output continuous coordinate values.

The four outputs represent the upper-left and lower-right corners of the bounding box:

$$[x_{min}, y_{min}, x_{max}, y_{max}]$$

These coordinates define a rectangle around the cat's face.

### Why remove the original classification head?

A classification head answers a question such as, "Which category does this image belong to?" A bounding-box regression head answers a different question: "Where is the face located in this image?"

The original classification output therefore does not match the output required for this task. Replacing it with a custom head allows the network to learn a mapping from image features to bounding-box coordinates.

The intermediate fully connected layers provide additional learnable transformations between the extracted features and the four coordinate predictions. The paper specifies the 128- and 64-unit layers, although it does not provide a separate justification for those exact widths. I therefore interpret them as the authors' chosen architecture rather than as uniquely optimal layer sizes.

This approach preserves the useful visual representations learned by the retrained backbone while adapting the model to a more specialized computer-vision task.

## 4. My Initial Experiment: Smooth L1 Loss

For my first experiment, I trained the face detector using Smooth L1 loss, also known as Huber loss.

Smooth L1 is commonly used for regression problems. It behaves approximately quadratically for small prediction errors and linearly for larger errors, making it less sensitive to large errors than a purely squared-error loss.

In my implementation, I used the following settings:

| Parameter | Setting |
|---|---|
| Model | EfficientNetV2-S with pretrained weights |
| Input resolution | 224 × 224 pixels |
| Training / validation / test split | 75% / 15% / 10% |
| Loss function | Smooth L1 (Huber) loss |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Training duration | 2 epochs |

I resized the input images to 224 $$\times$$ 224 pixels, following the resolution used in the research paper's face-detection stage. I initially avoided additional image transformations to keep the preprocessing and training process simple and computationally manageable.

The dataset was divided into training, validation, and test sets. The training set was used to update the model parameters, while the validation set provided feedback on performance during training. The test set was intended for evaluation on data not used for training or validation.

One practical limitation was the available GPU resources. I could only train the initial model for two epochs, so I treated the result as a first experiment rather than an exhaustive optimization of the model.

After these two epochs, the validation loss had decreased to approximately **0.0013.**

At first glance, this appeared promising. I then tested the model on additional cat images and visualized its predicted bounding boxes against the reference boxes. The initial predictions looked reasonably good in several examples.

However, the visual results revealed a limitation that the validation loss alone did not make obvious.

## 5. The Problem: A Good-Looking Loss Wasn't the Whole Story

When I inspected the predicted bounding boxes more closely, I noticed that some boxes did not include the entire cat's face. In particular, the predicted box could be too tight around the face and miss important features such as the ears or mouth.

For this project, that was a meaningful problem. The goal was not simply to locate the general area of a face; the crop needed to preserve the facial features required by the subsequent landmark-detection models.

Why didn't a seemingly low validation loss reveal this problem?

The key is that **validation loss and bounding-box overlap measure different aspects of performance**. Smooth L1 loss measures coordinate errors according to its own penalty function. Intersection over Union (IoU), on the other hand, measures how much the predicted box overlaps with the reference box.

A model can achieve a low loss value without consistently producing boxes that include every important facial feature. The numerical value of the loss also depends on how the coordinates are represented and scaled, so it should not be interpreted in isolation.

This was a useful lesson from the experiment: evaluating a bounding-box detector requires more than checking the training curve. I needed to inspect the predictions visually and assess how closely the predicted boxes matched the reference boxes.

The first experiment gave me a workable starting point, but I wanted to improve the overlap between the predicted and reference boxes.

## 6. Experimenting with CIoU Loss

To address the bounding-box problem, I experimented with **Complete Intersection over Union (CIoU) loss**.

IoU is calculated as the area of intersection between the predicted and reference boxes divided by the area of their union:

$$ IoU =
\frac{Area(B_p \cap B_{gt})}
{Area(B_p \cup B_{gt})} $$

Here, $$B_p$$ is the predicted bounding box and $$B_{gt}$$ is the ground-truth bounding box.

IoU provides and intuitive measure of localization quality: a larger value means that the two boxes overlap more closely.

CIoU extends this idea by accounting not only for overlap but also for the distance between the box centers and the difference in their aspect ratios. These additional terms provide the model with more information about how its predicted box differs from the reference box. If you're interested in learning more about CIoU or other loss functions used in bounding box regression, please check out my post on [Loss Functions for Object Detection](https://seyong2.github.io/portfolio/docs/data-science/computer-vision/iou-losses.html).

This made CIoU an interesting alternative for my task. Since the issue was that some predicted boxes did not cover the face properly, I wanted to explore a loss function more directly related to bounding-box geometry.

I trained a second model using CIoU loss for 15 epochs and compared its predictions with those from the original Smooth L1 model.

It is important to mention that the two models were trained for different numbers of epochs: **two for Smooth L1 and 15 for CIoU**. This was not a controlled experiment in which only the loss function changed. Consequently, the results show how the two documented experiments performed, but they cannot establish that the loss function alone caused the difference.

## 7. Comparing the Results

I evaluated the predicted bounding boxes using IoU, comparing each model's predictions with the ground-truth boxes.

The results recorded in my original experiments were:

| Model                             | Training duration | Average IoU |
| --------------------------------- | ----------------: | ----------: |
| EfficientNetV2-S + Smooth L1 loss |          2 epochs |      0.7521 |
| EfficientNetV2-S + CIoU loss      |         15 epochs |      0.8212 |

The CIoU experiment achieved a higher average IoU than the initial Smooth L1 experiment. The difference was approximately 0.0691, or 6.91 percentage points.

Visual comparison corroborates this result: the CIoU model generated more accurately overlapping bounding boxes, while the Smooth L1 model produced predictions that were too constricted around the face. To visualize this variance, several example images are provided below. Here, the ground truth box is marked in blue, the CIoU prediction in red, and the Smooth L1 prediction in green. The images clearly demonstrate that the green boxes (Smooth L1) often clip essential parts of the cat's face, potentially introducing errors in subsequent processing stages.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ac94002b-5403-42e2-a74a-e4a73b41d8a3" width="400">
  <img src="https://github.com/user-attachments/assets/064a93da-336a-4389-bf98-7c6b85090d66" width="400">
</p>

These results supported continuing with the CIoU model for the next stage of my pipeline. However, I would not interpret them as proof that CIoU is inherently superior to Smooth L1 for this task. The unequal training durations are an important limitation, and a fairer comparison would require training both models under comparable conditions.

I did not rerun the experiments to control for this difference, so I have retained the results from the original project rather than presenting new or unverified measurements.

## 8. What I learned

This experiment taught me a few practical lessons about building computer-vision models.

**First, transfer learning can provide a useful starting point.** Using pretrained EfficientNetV2-S weights meant that I could adapt an existing visual feature extractor rather than build a model from scratch. The classification head still needed to be replaced because the target task was bounding-box regression.

**Second, the evaluation metric needs to match the problem.** A low validation loss was encouraging, but it did not guarantee that the predicted face crop would retain all the features needed downstream. Visual inspection and IoU helped me identify this limitation.

**Third, experiments need to be interpreted in context.** The CIoU experiment achieved a higher average IoU, but the models were trained for different numbers of epochs. Recording that limitation is important when drawing conclusions from the results.

With the face detector in place, I could move on to the next stage: locating individual facial regions within the cropped face.

## 9. What's Next?

Detecting the face was only the first step. To extract useful facial information, I also needed to locate specific regions, such as the eyes, ears, and mouth, before predicting their landmarks.

In the next article, I'll explore how I approached facial region detection and how the pipeline progressed from a face bounding box to a structured set of facial landmarks.

---
#### Resources
- Martvel, G., Shimshoni, I., and Zamansky, A. (2024). [Automated Detection of Cat Facial Landmarks](https://doi.org/10.1007/s11263-024-02006-w). *International Journal of Computer Vision*, 132, 3103–3118.
- [TorchVision: EfficientNetV2-S](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html)
- [CatFLW dataset](https://www.kaggle.com/datasets/georgemartvel/catflw/data)
