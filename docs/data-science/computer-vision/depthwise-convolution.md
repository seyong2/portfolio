---
title: Depthwise Separable Convolution
parent: Computer Vision
nav_order: 1
layout: default
---

Depthwise separable convolution is a variation of standard convolution used primarily in deep learning models to *reduce the computational complexity of convolutional layers*. It is especially common in lightweight architectures like MobileNet.

In a traditional 2D convolution, you apply a single convolutional filter (or kernel) to all the input channels at once and produce a single output channel. However, in **depthwise separable convolution**, the process is split into two parts: 

1. **Depthwise convolution**: Instead of applying one filter to all input channels, a separate filter is applied to each input channel independently. Each channel is convolved with its corresponding filter, producing an output channel for each input channel. This captures spatial features but doesn't mix information across channels.
  
2. **Pointwise convolution**: To combine the output from the depthwise convolution, a 1 $$\times$$ 1 convolution (pointwise convolution) is typically used to linearly combine the output channels. This operation is used to mix information from different channels.

### Why use depthwise convolution?

The main advantage is that it drastically reduces the number of parameters and computations, making models more efficient. Instead of convolving across all input channels simultaneously, depthwise convolution processes each channel individually, which simplifies the computation.

Let's take a simple example to better understand how depthwise convolution helps reduce computational complexity. For this, we'll compare the number of multiplications required in standard convolution versus depthwise convolution. Suppose we have an input tensor with shape of 6 $$\times$$ 6 $$\times$$ 3 (height, width, and channels) and we want to apply a 3 $$\times$$ 3 convolution with 4 filters. We set the stride to 1 and the padding to 0.

### Standard convolution

Each filter has a size of 3 $$\times$$ 3 $$\times$$ 3 (because the filter must span all 3 input channels). The filter would slide across the input and produce one output channel (4 $$\times$$ 4) as can be seen below plot.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e67fe61b-e7f7-4635-807c-491d87765745">
</p>

Each filter performs 3 $$\times$$ 3 $$\times$$ 3 = 27 multiplications per pixel in the input. The filter is then slid across the input, covering a 4 $$\times$$ 4 area (along the width and height), resulting in 4 $$\times$$ 4 convolution operations. Therefore, for each filter, we perform 4 $$\times$$ 4 $$\times$$ 27 = 432 multiplications. With 4 filters, total number of operations for the entire feature map is 4 $$\times$$ 432 = 1,728. This is computationally expensive, especially as the number of input channels and filters increases.

### Depthwise separable convolution

- **Depthwise convolution**

Instead of applying 3 $$\times$$ 3 $$\times$$ 3 filter across all input channels, we apply a separate 3 $$\times$$ 3 filter to each channel individually. Since the input has 3 channels, we apply 3 filters (one per channel), and each filter is 3 $$\times$$ 3.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f0a4527b-ebb9-4468-bd53-2d948764a03b">
</p>

Each depthwise filter performs 3 $$\times$$ 3 = 9 multiplications per convolution operation. As always, the filter is slid across the input, so we perform 4 $$\times$$ 4 $$\times$$ 9 = 144 multiplications. Since there are 3 channels, the number of multiplications for the depthwise convolution is 3 $$\times$$ 144 = 432. 

- **Pointwise convolution**

After depthwise convolution, we perform pointwise convolution, which is simply a 1 $$\times$$ 1 convlution applied across all input channels. The input to the pointwise convolution is the 4 $$\times$$ 4 output from the depthwise step, with 3 channels. We apply 4 1 $$\times$$ 1 filters (one per output channel).

<p align="center">
  <img src="https://github.com/user-attachments/assets/3d480522-84bf-4d7a-b87b-7a229f27bac0">
</p>

Each 1 $$\times$$ 1 filter performs 1 multiplication per input channel, so each 1 $$\times$$ 1 filter performs 3 multiplications per convolution operation. Sliding through the input, we perform in total 4 $$\times$$ 4 $$\times$$ 3 = 48 multiplications. As we have 4 filters, the total number of multiplications for the pointwise convolution is equal to 48 $$\times$$ 4 = 192.

The total number of operations for depthwise separable convolution is the sum of the depthwise and pointwise operations:

$$ 432 \ (depthwise) + 192 \ (pointwise) = 624 \ multiplications. $$

By using depthwise separable convlution, we've reduced the computational cost by almost 3 times (1,728 vs. 624 multiplications).

This process drastically reduces the number of parameters and computations, making depthwise separable convolutions ideal for resource-constrained environments, such as mobile and embedded devices.
