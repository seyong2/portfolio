---
title: EfficientNet Explained
parent: Computer Vision
nav_order: 3
layout: default
---

In this post, based on the paper **EfficientNEt: Rethinking Model Scaling for Convolutional Neural Networks** by Mingxing Tan and Quoc V. Le, we'll explore EfficientNet and what distinguishes this neural network architecture from other convolutional neural networks (ConvNets). 

It's well known that larger ConvNets tend to achieve higher accuracy, often by scaling up models along one of three dimensions: depth, width, or input resolution. However, due to the over-parameterization of deep ConvNets, there have been efforts to compress models to gain efficiency, but often at the cost of accuracy.

This paper introduces a method that allows ConvNets to achieve improved accuracy by effectively scaling all three dimensions (width, depth, and input resolution) simultaneously. Additionally, it emphasizes that the effectiveness of model scaling is highly dependent on the baseline network, proposing a new baseline network, **EfficientNet**, which surpasses other existing ConvNets in both accuracy and efficiency.

## Model Scaling

Scaling up a ConvNet generally involves increasing the model’s depth, width, or input resolution. Each of these changes can bring specific benefits:

- **Depth** (adding layers): More layers enable the model to capture richer, more complex features, which can improve generalization on new tasks. However, adding depth can lead to issues like the vanishing gradient problem. Although techniques, like skip connections and batch normalization help, mitigate this, they may sometimes result in lower accuracy.

- **Width** (increasing filters per layer): Wider networks have more filters per layer, allowing them to capture a broader variety of features. This can be helpful for datasets with high variance, as the network can learn multiple representations at each layer. However, very wide but shallow networks may struggle to capture higher-level features.

- **Resolution** (higher input size): Increasing the input resolution helps capture finer details within the data. Higher resolutions generally improve accuracy, though the benefit diminishes at very high resolutions.

<p align="center">
  <img src="https://github.com/user-attachments/assets/94a6b8b0-97d6-4880-98bc-752ee3c04442">
</p>

Although scaling along multiple dimensions is possible, it often requires tedious manual tuning, as the optimal scaling depends on how each dimension interacts with the others and on specific resource constraints.

### Compound Scaling

Scaling multiple dimensions isn't independent—different scaling dimensions affect one another. For example, with higher-resolution images, the model’s depth should increase to capture broader context from larger receptive fields, while the width should increase to capture more diverse patterns within each layer.

To address this, the paper introduces a compound scaling method that balances all three scaling dimensions rather than focusing on one. As shown in the figure below, coordinating depth, width, and resolution scaling results in improved accuracy.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c21e09fe-a8ae-408c-b6d5-83d0ce3e6584">
</p>

The paper proposes a compounding scaling method that eliminates the need for manual tuning. It introduces a compound coefficient, $$\phi$$, that uniformly scales width, depth, and resolution as follows:

$$ depth:\ d=\alpha^{\phi}$$
$$ width: \ w=\beta^{\phi}$$
$$ resolution: \ r=\rho^{\phi}$$
$$ s.t. \ \alpha\cdot\beta^2\cdot\rho^2 \approx 2$$
$$ \alpha \ge 1,\ \beta \ge 1, \ \rho\ge 1 $$

In this equation, $$d$$, $$w$$, and $$r$$ are scaling coefficients for depth, width, and resolution, respectively. The compound coefficient $$\phi$$ controls how many additional resources are available for model scaling, while $$\alpha$$, $$\beta$$, and $$\rho$$ are constants (determined by a small grid search) that specify how resources are allocated across each dimension.

FLOPs (Floating Point Operations per Second) serve as a measure of computational complexity, representing the number of floating-point operations (like additions and multiplications) that a model performs to make predictions. FLOPs are proportional to $$d$$, $$w^2$$, and $$r^2$$, as doubling width or resolution increases width and height. EfficientNet constrains $$\alpha\cdot\beta^2\cdot\rho^2 \approx 2$$ to ensure FLOPs increase by $$2^{\phi}$$ for each new $$\phi$$ value.

## EfficientNet Architecture

A mobile-size baseline, EfficientNet, was developed using a multi-objective neural architecture search that optimizes both accuracy and FLOPs. The table below shows the baseline architecture for EfficientNet-B0.

<p align="center">
  <img src="https://github.com/user-attachments/assets/86e87a8b-872b-404e-9b98-0f08223fa36a">
</p>

The compound scaling method can be applied to EfficientNet-B0 in two steps:

1. Fix $$\phi=1$$ (assuming twice more resources are available) and perform a small grid search to find the optimal values for $$\alpha$$, $$\beta$$, and $$\rho$$. For EfficientNet-B0, the best values were $$\alpha=1.2, \ \beta=1.1, \ \rho=1.15$$, under the constraint $$\alpha\cdot\beta^2\cdot\rho^2 \approx 2$$.
  
2. After fixing $$\alpha$$, $$\beta$$, and $$\rho$$, scale up the baseline network with different $$\phi$$ values to obtain EfficientNet-B1 to B7.

To conclude, EfficientNet introduces a compelling approach to model scaling, optimizing depth, width, and resolution in a balanced, compound manner rather than individually. By using a scalable baseline architecture and a compound scaling method, EfficientNet demonstrates a significant improvement in accuracy and efficiency compared to traditional ConvNets. This advancement not only enhances performance on benchmark datasets like ImageNet but also offers broader applicability in transfer learning tasks, showing that efficiency and accuracy can indeed go hand-in-hand. EfficientNet serves as a powerful model architecture for tasks requiring high accuracy with manageable computational resources, paving the way for more resource-efficient deep learning applications across diverse domains.

#### Resources
- [EfficientNEt: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946)
