---
title: Canny Edge Detection
parent: Computer Vision
nav_order: 6
layout: default
---

Edge detection is a fundamental technique in computer vision, used to identify edges—defined as curves in a digital image where brightness changes sharply or, more formally, where discontinuities occur. Among various methods, the Canny edge detector, developed by John F. Canny in 1986, remains a state-of-the-art operator. It uses a multi-stage algorithm to robustly detect a wide range of edges in images.

In this notebook, we will explore how the Canny edge detection algorithm works using an example image of a cat. Let’s begin by importing the necessary Python libraries and loading the sample image.

```python
# Import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
```

```python
# Load the image
img = Image.open('/Users/syryoo/Downloads/thor2.jpg')
```

```python
# Display the image
plt.imshow(img)
plt.axis('off')
plt.show()
```

<p align="center">
  <img width="511" height="389" src="https://github.com/user-attachments/assets/bdc8c632-b5d9-49d7-8fd9-1b7553dd143d">
</p>

![thor2]()

The Canny edge detection algorithm consists of five main steps:

1. Apply a Gaussian filter to smooth the image and reduce noise.
2. Compute the intensity gradients of the image.
3. Perform non-maximum suppression to eliminate spurious responses to edge detection.
4. Apply double thresholding to identify potential edges.
5. Use edge tracking by hysteresis to finalize the edge map, retaining only strong edges and those connected to them.
6. Before beginning this process, the image must be converted to grayscale. This simplifies the computation by reducing the image to a single intensity channel, while preserving essential structural information. To perform the conversion, we use the standard luminance formula:

$$L=0.299R+0.587G+0.114 B$$

These coefficients reflect human visual perception—our eyes perceive green as brighter than red, and red as brighter than blue.

```python
# Convert to grayscale
gray_img = img.convert('L')
plt.imshow(gray_img, cmap='gray')
plt.axis('off')
plt.show()
```

<p align="center">
    <img width="511" height="389" src="https://github.com/user-attachments/assets/ad99a8b1-af7a-44df-8773-879b306ce15e">
</p>

## 1. Noise Reduction - Gussian Blur

Since edge detection is highly sensitive to noise, it is crucial to first reduce noise in the image to avoid false edge responses. To smooth the image, a Gaussian filter kernel is convolved with it. This step slightly blurs the image, mitigating the impact of noise while preserving important structural features. The equation for a Gaussian filter kernel of size $(2k+1)\times(2k+1)$ is:

$$
H_{ij} = \frac{1}{2\pi\sigma^2}exp(\frac{(i-(k+1))^2+(j-(k+1))^2}{2\sigma^2}); 1\le i,j\le(2k+1)
$$

The choice of the Gaussian kernel size significantly influences the detector's performance. A larger kernel reduces sensitivity to noise but may also introduce a localization error, making edges appear less precisely defined. In practice, a 5 $\times$ 5 works well in many cases, though the optimal size may vary depending on the specific application.

```python
def gaussian_kernel(size=5, sigma=1.0):
    """Returns a 2D Gaussian kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def gaussian_filter(image, kernel):
    """Applies Gaussian filter to the image."""
    filter_dim = kernel.shape[0]
    pad_size = filter_dim // 2
    padded_image = np.pad(image, pad_size, mode='constant')
    
    # Create a view: all 3x3 windows
    shape = (image.shape[0], image.shape[1], filter_dim, filter_dim)
    strides = (padded_image.strides[0], padded_image.strides[1],
               padded_image.strides[0], padded_image.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded_image, shape=shape, strides=strides)
    
    # Apply the kernel to each window
    g = np.sum(windows * kernel, axis=(2, 3))
    g = g / np.max(g) * 255
    return g
```

```python
gray_img_arr = np.array(gray_img)
kernel = gaussian_kernel(size=5, sigma=1.0)
smoothed_img = gaussian_filter(gray_img_arr, kernel)

plt.imshow(smoothed_img, cmap='gray')
plt.axis('off')
plt.show()
```

<p align="center">
    <img width="511" height="389" src="https://github.com/user-attachments/assets/1ad2ef73-7b2c-4624-b49a-f7ae62f55904">
</p>

## 2. Gradient Calculation - Sobel Operator

To create an image that emphasizes edges, we apply the Sobel operator. Since edges correspond to rapid change in brightness, we need a way to measure how quickly the intensity varies across the image. This is achieved by estimating the gradient of image intensity, which is why the Sobel method is known as a gradient calculation technique. 

The Sobel operator uses two convolution kernels to estimate the gradient; one for the horizontal direction ($G_x$) and one for the vertical direction ($G_y$). These approimate the partial derivatives of the image:

$$
G_x = \frac{\partial I}{\partial x}, G_y=\frac{\partial I}{\partial y}
$$

Once the gradients are computed, we can combine them to obtain the gradient magnitude ($||\nabla I||$), which indicates the strength of the edge, and the gradient direction ($\theta$), which tells us the orientation of the edge.

$$
||\nabla I|| = \sqrt{G_x^2+G_y^2}
$$

$$
\theta = arctan(\frac{G_x}{G_y})
$$

Pixels with large gradient magnitudes are likely to be edges, and the direction $\theta$ helps determine their orientation.

```python
def sobel_edge_detection_vectorized(image):
    """ Computes Sobel edge detection using vectorized operations. """
    # Define Sobel filters
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    # Apply padding
    filter_dim = sobel_x.shape[0]
    pad = filter_dim // 2
    padded_image = np.pad(image, pad_width=pad, mode='constant')

    # Create a view: all 3x3 windows
    shape = (image.shape[0], image.shape[1], filter_dim, filter_dim)
    strides = (padded_image.strides[0], padded_image.strides[1],
               padded_image.strides[0], padded_image.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded_image, shape=shape, strides=strides)
    
    # Compute gradients (broadcasted convolution)
    gx = np.sum(windows*sobel_x, axis=(2, 3))
    gy = np.sum(windows*sobel_y, axis=(2, 3))

    # Gradient magnitude
    edges = np.sqrt(gx**2 + gy**2)
    edges = edges / np.max(edges) * 255
    theta = np.arctan2(gy, gx)
    
    return (edges, theta)
```

```python
edges, thetas = sobel_edge_detection_vectorized(smoothed_img)

plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()
```

<p align="center">
    <img width="511" height="389" src="https://github.com/user-attachments/assets/6ef8021a-7238-4bb3-8e65-9b05195c648c">
</p>

## 3. Non-Maximum Suppression

Since the initial edge detection often produces edges with varying thickness, we apply non-maximum suppression to thin them and retain only the pixels corresponding to the sharpest intensity changes.

The algorithm works on the gradient magnitude and direction image as follows:

1. For each pixel, compare its edge strength (gradient magnitude) with the strengths of the two neighboring pixels along the gradient directions (positive and negative).
2. If the current pixel has the highest magnitude among the three, it is kept as part of an edge: otherwise, it is suppressed (set to zero).

In our example, we first convert the gradient direction $\theta$ from radians to degrees. Any negative angles (measured clockwise from the positive $x$-axis) are rotated counter-clockwise so that all angles range fall within the range 0° and 180°.

We then categorize each pixel's angle into one of four directions:

| Angle Range             | Direction                 | Pixels to Compare  |
| ----------------------- | ------------------------- | ------------------ |
| 0°–22.5° or 157.5°–180° | Horizontal                | Left & Right       |
| 22.5°–67.5°             | Diagonal (Right ↘ Left ↖) | Diagonal neighbors |
| 67.5°–112.5°            | Vertical                  | Up & Down          |
| 112.5°–157.5°           | Diagonal (Left ↘ Right ↖) | Diagonal neighbors |

Once the appropriate neighboring pixels are identified, we suppress all non-maximum values, effectively thinning the edges to a one-piel width.

```python
def non_maximum_suppression(image, thetas):
    """Applies non-maximum suppression to the edge-detected image."""
    result = np.zeros((image.shape[0], image.shape[1]))
    angles = thetas * 180.0 / np.pi
    angles[angles < 0] += 180
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            try:
                p = image[i, j]
                q = 255
                r = 255
                if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                    q = image[i, j-1]
                    r = image[i, j+1]
                elif (22.5 <= angles[i, j] < 67.5):
                    q = edges[i+1, j-1]
                    r = image[i-1, j+1]
                elif (67.5 <= angles[i, j] < 112.5):
                    q = image[i+1, j]
                    r = image[i-1, j]
                elif (112.5 <= angles[i, j] < 157.5):
                    q = image[i-1, j-1]
                    r = image[i+1, j+1]
                
                if (p >= q) and (p >= r):
                    result[i, j] = p
                else:
                    result[i, j] = 0
            except IndexError:
                pass
    return result
```

```python
edges_suppressed = non_maximum_suppression(edges, thetas)

plt.imshow(edges_suppressed, cmap='gray')
plt.axis('off')
plt.show()
```

<p align="center">
    <img width="511" height="389" src="https://github.com/user-attachments/assets/a7a292c8-06de-4f37-ae07-914a7bcbfbb5" >
</p>

## 4.Double Thresholding

To distinguish real edges from noise, we apply double thresholding, which classifies each pixel into one of three categories: 

- Strong pixels: High intensity pixels that are very likely to be part of a true edge.
- Weak pixels: Pixels with moderate intensity- too low to be considered strong, yet too high to be discared immediately.
- Non-relevant pixels: Low intensity pixels that are treated as noise and completely suppressed.

Pixels with intensities above the high threshold are marked as strong, while those between the high and low thresholds are marked as weak. Pixels below the low threshold are discarded.

Weak pixels require further analysis. This is done using hysteresis, which determines whether a weak pixel should be treated as a real edge.

The values for the high and low thresholds are chosen empirically and depend on the characteristics of the input image.

```python
def double_thresholding(image, high_threshold_ratio, low_threshold_ratio):
    high_threshold = np.max(image) * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    result = np.zeros((image.shape[0], image.shape[1]))

    strong = 255
    weak = 25

    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))
    non_relevant_i, non_relevant_j = np.where(image < low_threshold)

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    result[non_relevant_i, non_relevant_j] = 0

    return result
```

```python
result = double_thresholding(edges_suppressed, high_threshold_ratio=0.09, low_threshold_ratio=0.05)

plt.imshow(result, cmap='gray')
plt.axis('off')
plt.show()
```

<p align="center">
    <img width="511" height="389" src="https://github.com/user-attachments/assets/89ff5fa8-0246-443b-9cbf-b98dbc58a5f9">
</p>

## 5. Edge Tracking by Hysteresis

After double thresholding, we are left with strong pixels (definite edges) and weak pixels (potential edges). However, weak pixels can either represent real edges or noise — so we need a way to decide which ones to keep.
Hysteresis handles this by examining the connectivity of weak pixels:

A weak pixel is promoted to a strong edge pixel if it is connected to at least one strong pixel in its neighborhood (typically a 3 $\times$ 3 window). Otherwise, it is suppressed as noise.

This step ensures that only meaningful, continuous edges are retained. Weak pixels that are isolated or surrounded by non-relevant pixels are discarded, while those that form part of a coherent edge structure are preserved.
The result is a clean, final edge map that captures continuous edges while minimizing false detections.

```python
def hysteresis(image):
    """ Applies hysteresis to the double-thresholded image. """
    weak_i, weak_j = np.where(image == 25)

    for i, j in zip(weak_i, weak_j):
        try:
            region = image[i-1:i+1, j-1:j+1]
            if 255 in region:
                image[i, j] = 255
            else:
                image[i, j] = 0
        except IndexError:
            pass

    return image
```

```python
result_hysteresis = hysteresis(result)
plt.imshow(result_hysteresis, cmap='gray')
plt.axis('off')
plt.show()
```

<p align="center">
    <img width="511" height="389" src="https://github.com/user-attachments/assets/6884d1ff-4524-40da-95c1-b845cedd4156">
</p>

As we've seen through the example, the Canny edge detection algorithm is capable of reliably identifying clear and meaningful edges in images. I hope this notebook helped you better understand how it works.

