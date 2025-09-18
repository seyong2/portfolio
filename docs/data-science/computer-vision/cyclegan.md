---
title: Horse to Zebra through CycleGAN
parent: Computer Vision
nav_order: 5
layout: default
---

CycleGAN is a deep learning model that can transform images from one domain to another—for example, turning horses into zebras. Introduced in the paper **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks**, CycleGAN learns to capture the unique characteristics of one image collection and translate them into another, all without requiring paired training examples. In this article, we'll explore how the model works under the hood.

## Vanilla GAN

Suppose we have two sets of images: one from domain $$ X $$ (e.g., horses) and another from domain $$ Y $$ (e.g., zebras). Our goal is to train a model $$ G: X \rightarrow Y $$ that takes an image $$ x \in X $$ and generates a new image $$ y_{hat} = G(x) $$ that looks like it came from domain $$ Y $$. To achieve this, the generator learns to create images that are so realistic that a second model, called the discriminator, cannot tell them apart from real images $$ y \in Y $$. In the end, the optimal $$ G $$ should translate images from $$ X $$ into a new domain $$ Y_{hat} $$ whose distribution is indistinguishable from $$ Y $$.

However, this approach has some limitations. The way the generator learns to translate images from one domain to another does not guarantee that the translation is meaningful. This is because there are infinitely many ways for $$ G $$ to produce images whose distributions are indistinguisable from real ones. For instance, if we want to map a horse image to a zebra-like image, one generator might correctly add black-and-white stripes to the hosr, while another might alter the background or place random stripes elsewhere. Both results may fool the discriminator, but only the first preserves the structure of the input horse. In contrast, the second type of output makes the horse look like a zebra in an inconsistent or strange way.

In addition, optimizing the adversarial objective alone is quite challenging. Standard training procedures often lead to the well-known problem of mode collapse, where different input images produce the same output image and learning stalls. This happens when the generator discovers a shortcut: a single type of output that consistently fools the discriminator. As a result, training becomes unstable, and the generator fails to learn meaningful mappings between domains.

These limitations highlight the need to add more structure to our objective. One key idea is cycle consistency: if we translate an image from domain $$ X $$ to domain $$ Y $$ (e.g., a horse to a zebra), and then translate it back from $$ Y $$ to $$ X $$, we should recover the original image (the same horse we started with).

## CycleGAN

CycleGAN uses two generators; a generator $$ G: X \rightarrow Y $$, which translates images from $$ X $$ to $$ Y $$, and $$ F: Y \rightarrow X $$, which translates in the opposite direction. These two models are trained together with an additional cycle consistency loss, which enforces $$ F(G(x)) \approx x $$ and $$ G(F(y)) \approx y $$. By combining this loss with the adversarial losses on domains $$ X $$ and $$ Y $$, CycleGAN achieves unpaired image-to-image translation.

### Formulation

Given training samples $$ {x_i}^N_{i=1} $$ and $$ {y_j}^N_{j=1} $$ from two domains $$ X $$ and $$ Y $$, we denote their distributions as $$ x \sim p_{data}(x) $$ and $$ y \sim p_{data}(y) $$. CycleGAN uses two adversarial discriminators, $$ D_X $$ and $$ D_Y $$: $$ D_X $$ tries to distinguish between real images $$ {x} $$ and translated images $$ {F(y)} $$, while $$ D_Y $$ discriminates between real images $$ {y} $$ and generated images $$ {G(x)} $$.

The adversarial losses for matching the distribution of generated images to the target domain are:

$$ L_{GAN}(G,D_Y,X,Y)=E_{y~p_{data}(y)}[log D_Y(y)]+E_{x~p_{data}(x)}[log (1-D_Y(G(x)))] $$ for $$ G: X \rightarrow Y $$

$$ L_{GAN}(F,D_X,Y,X)=E_{x~p_{data}(x)}[log D_X(x)]+E_{y~p_{data}(y)}[log (1-D_X(F(y)))] $$ for $$ F: Y \rightarrow X $$

In the adversarial framework, the generators aim to minimize these losses, while the discriminators aim to maximize them:

$$ min_{G}max_{D_Y} L_{GAN}(G,D_Y,X,Y), min_{F}max{D_X} L_{GAN}(F,D_X,Y,X) $$ 

CycleGAN introduces an additional objective called the cycle consistency loss, which prevents the two learned mappings, $$ G $$ and $$ F $$, from contradicting each other.

Adversarial training alone can, in principle, learn mappings where $$ G $$ produces outputs distributed like domain $$ Y $$ and $$ F $$ produces outputs distributed like domian $$ X $$. However, if the networks have enough capacity, they could simply map inputs to arbitrary permutations of the target domain. In that case, the generated distribution would still match the target distribution, but the mapping from an individual input $$ x_i $$ to a specific output $$ y_i $$ would be meaningless. In other words, adversarial loss by itself cannot guarantee that the learned translation preserves the content of each input image.

To address this issue, CycleGAN enforces cycle consistency. The idea is that translating an image to the other domain and then back again should recover the original image. For every image $$ x $$ from domain $$ X $$, the round-trip translation should bring it back to the same $$ x $$, that is, $$ x \rightarrow G(x) \rightarrow F(G(x)) \approx x $$ which is known as forward cycle consistency. Likewise, for every image $$ y $$ from domain $$ Y $$, applying $$ F $$ followed by $$ G $$ should return the original $$ y $$: $$ y \rightarrow F(y) \rightarrow G(F(y)) \approx y $$, a condition referred to as backward cycle consistency. This principle is captured by the cycle consistency loss, which explicitly penalizes discrepancies between the original and reconstructed images: 

$$ L_{cyc}(G, F)= E_{x~p_{data}(x)}[||F(G(x))-x||_1] + E_{y~p_{data}(y)}[||G(F(y))-y||_1] $$

where $$ ||\cdot||_1 $$ is L1 norm, measuring the absolute difference between the input and its reconstruction.

Finally, the full objective of CycleGAN combines the adversarial losses with the cycle consistency loss:

$$ L(G, F, D_X, D_Y)= L_{GAN}(G, D_Y, X, Y) + L_{GAN}(F, D_X, Y, X) + \lambda L_{cyc}(G, F), $$
where $$ \lambda $$ is a hyperparameter that balances the importance of the cycle consistency loss relative to the adversarial losses.

The overall training problem is then formulated as:

$$ G^*, F^* = arg min_{G,F} max_{D_X, D_Y} L(G, F, D_X, D_Y). $$

### Implementation

#### Generator

The generator networks in CycleGAN adopt the architecture proposed by Johnson et al., which had previously shown strong performance in tasks such as neural style transfer and image super-resolution. Both tasks rely heavily on the ability of deep convolutional networks to learn rich visual representations, making this architecture a natural choice for image-to-image translation.

The generator is composed of three initial convolutional layers, followed by a sequence of residual blocks, and finally two fractionally strided convolutions with stride $$ \frac{1}{2} $$ that upsample the feature maps. The output is then mapped back to the RGB space through a final convolutional layer. In practice, the authors used six residual blocks when training on 128 $$ \times $$ 128 images, and nine residual blocks for images of resolution 256 $$ \times $$ or higher. Instance normalization is employed throughout the generator, in line with the design choices of Johnson et al.

The network contains three convolutions, several residual blocks, two fractionally-strided convolutions with stride 1/2, and one convolution that maps features to RGB. The authors used 6 blocks for 128 /times 128 images and 9 blocks for 256 /times 256 and higher-resolution training images. Similar to Johnson et al., they use instance normalization. 

Residual blocks play a crucial role in this architecture. As networks become deeper, they can theoretically learn more complex functions, but in practice they often face optimization challenges such as vanishing gradients or degraded accuracy. Residual connections address this problem by allowing the network to focus on learning changes relative to the input rather than having to relearn the entire mapping. Formally, instead of directly approximating a function $$ f(x) $$, a residual block learns a residual mapping $$ g(x)=f(x)−x $$. The output of the block is then computed as $$ f(x)=g(x)+x $$. When the residual $$ g(x) $$ is small-meaning the desired output is already close to the input-the block behaves almost like an identity function, passing its input unchanged. This structure not only stabilizes training but also improves the representational capacity of the network by enabling deeper models.

The upsampling layers in the generator are implemented using fractionally-strided convolutions, sometimes referred to as transposed convolutions or deconvolutions. These layers increase the spatial resolution of the feature maps, effectively reversing the downsampling performed by the eariler convolutional layers. With a stride of $$ \frac{1}{2} $$, each input pixel is effectively expanded, doubling the spatial resolution of the feature maps by inserting zeros between elements and applying a convolution with stride 1. This allows the generator to progressively reconstruct high-resolution images from compact feature representations.

To stabilize training further, CycleGAN employs instance normalization, a technique commonly used in style transfer and image generation. Unlike batch normalization, which normalizes activations across a mini-batch, instance normalization normalizes each sample independently across its spatial dimensions. This ensures that each image is treated consistently, without being influenced by other images in the batch, which is especially useful for style transfer and domain translation tasks.

#### Discriminator

FThe discriminator networks in CycleGAN follow a different design, known as the 70 × 70 PatchGAN architecture. Instead of classifying an entire image as real or fake, the PatchGAN discriminator operates on overlapping image patches of size 
70 $$ \times $$ 70. Each patch is classified independently, and the final output is a grid of predictions that collectively describe whether different parts of the image look realistic. This patch-level approach forces the generator to focus on producing high-quality local details such as textures, edges, and small structures, rather than only capturing the overall image structure. Additionally, because PatchGAN is fully convolutional, it can be applied to images of arbitrary size while using significantly fewer parameters than a full-image discriminator.

### Conclusion

In summary, CycleGAN provides a powerful framework for unpaired image-to-image translation by combining adversarial training with cycle consistency. This approach allows the model to generate realistic images in the target domain while preserving the content of the original images. By leveraging carefully designed generator and discriminator architectures, CycleGAN can handle a wide variety of translation tasks without requiring paired training data. Its success has inspired numerous extensions and applications in both research and practical image editing tasks.

#### Resources
- [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661)
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593)
- [Residual Blocks](https://rodriguesthiago.me/posts/residual_blocks/)
- [Instance Normalization vs Batch Normalization](https://www.geeksforgeeks.org/deep-learning/instance-normalization-vs-batch-normalization/)