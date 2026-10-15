---
title: Identifying 5 Cat Face Regions
parent: MoodMeow - Cat Emotion Recognition
nav_order: 5
layout: default
---

After detecting and cropping a cat's face, the next step was to identify the locations of its individual facial regions. These regions would serve as reference points for the following stage, where I would predict the detailed facial landmarks needed to extract facial characteristics.

Rather than predicting all 48 facial landmarks directly from the cropped image in a single step, I divided the task into smaller steps. First, I trained a model to predict the centers of five facial regions. These predicted centers would then help locate each region for more detailed landmark detection.

The five regions in my implementation were:

* Left eye
* Right eye
* Nose and mouth area
* Left ear
* Right ear

This intermediate step provided a structured way to move from a face-level crop to more localized facial information.

### 1. Computing Region Centers

To train a region detector, I needed target coordinates for the center of each region. Fortunately, the CatFLW dataset already provided annotated facial landmarks, so I could derive the region centers from those existing annotations instead of manually labeling five additional points for every image.

The landmarks were grouped according to their corresponding facial regions. For each region, I calculated the average of its landmark coordinates:

$$x_c = \frac{1}{n}\sum_{i=1}^{n}x_i, \qquad y_c = \frac{1}{n}\sum_{i=1}^{n}y_i$$

Here, $$n$$ is the number of landmarks assigned to a region, and $$x_i$$ and $$y_i$$ are the coordinates of each landmark. The resulting $$(x_c, y_c)$$ pair presents the region's center.

In my implementation, the landmarks were grouped as follows:

| Facial region | Number of landmarks |
|---|---:|
| Left eye | 8 |
| Right eye | 8 |
| Left ear | 5 |
| Right ear | 5 |
| Nose area | 22 |
| **Total** | **48** |

By averaging the landmarks within each region, I converted the original 48 annotated points into five target center points. Each image therefore had five region centers, represented by ten coordinate values in total: one $$x$$-coordinate and one $$y$$-coordinate per region.

This transformation made the target for the region-detection model simpler than predicting all 48 landmarks at once. The centers were not intended to replace the detailed landmarks; they provided intermediate locations that could guide the next stage of the pipeline.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ceb8173-64aa-45cf-b3e5-1f58ecfb3104" width="400">
</p>

### 2. Training the Region Detector

For the region detector, I used a pretrained EfficientNetV2 model, following the transfer-learning approach used in my face-detection stage.

The model took a cropped cat-face image as input and predicted the coordinates of the five region centers. Since these outputs were continuous numerical values rather than class labels, I treated the task as a regression problem.

#### Normalizing the coordinates

Before training, I normalized the target coordinates. This put the coordinates on a consistent numerical scale rather than using raw pixel values.

For each image, I normalized the landmark coordinates by dividing the horizontal coordinates by the image width and the vertical coordinates by the image height:

$$x_{\text{norm}} = \frac{x}{W}, \qquad y_{\text{norm}} = \frac{y}{H}
$$

where $$W$$ and $$H$$ are the image width and height, respectively. This maps the coordinates to a scale between 0 and 1, assuming the original coordinates lie within the image boundaries. It also makes the target coordinates independent of the image's pixel dimensions.

Coordinate normalization is useful for regression tasks involving image positions because the model can learn to predict values on a common scale. It also avoids having the numerical magnitude of the targets depend directly on the image's pixel dimensions.

The model was therefore trained to predict normalized coordinates, which could subsequently be converted back to image coordinates when locating the regions.

#### Data augmentation

I also applied random rotations to the training images. Cat faces do not always appear perfectly upright, so rotation augmentation was intended to expose the model to different face orientations during training.

An important part of this process is ensuring that the target coordinates are transformed consistently with the image. When an image is rotated, the corresponding region centers must be rotated as well; otherwise, the training image and its target coordinates would no longer match.

This augmentation was intended to help the model learn a more robust relationship between the appearance of a cat's face and the locations of its facial regions.

#### Loss function

I trained the model using Mean Squared Error (MSE) loss. MSE measures the squared differences between predicted and target coordinates:

$$MSE = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2$$

Here, $$\hat{y}_i$$ is a predicted coordinate, $$y_i$$ is the corresponding target coordinate, and $$N$$ is the number of values included in the average.

Because the task was to predict coordinates, MSE provided a way to penalize predictions that deviated from the target positions. Larger errors receive a disproportionately larger penalty because the differences are squared.

During training, the model adjusted its parameters to reduce the difference between the predicted region centers and the centers derived from the annotated landmarks.

### 3. Evaluating the Predictions

After training, I inspected the model's predictions on two test images. In the visualizations below, the blue points represent the ground-truth region centers, while the red points represent the centers predicted by the model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/78718d24-6ea2-4a85-9b45-ac2e43875189" width="400">
  <img src="https://github.com/user-attachments/assets/6176c360-965e-4f22-b62d-9ea208d9dfa0" width="400">
</p>

In these examples, the predicted centers are close to the corresponding ground-truth centers. This provided a qualitative indication that the model could identify the approximate locations of the five facial regions in these images.

The model's recorded validation MSE was **0.0028**.

Because the target coordinates were normalized before training, this value represents error on the normalized coordinate scale rather than directly in pixels. However, the loss value alone is not enough to determine whether the localization accuracy was good. Its interpretation depends on the coordinate normalization, the averaging convention used to calculate the loss, and the distribution of prediction errors.

For that reason, I would not treat 0.0028 as proof of high accuracy on its own. The visualizations provide useful examples of the model's behavior, but two images cannot establish how consistently it performs across the entire test set. A more complete evaluation would require a recorded test-set metric, such as average coordinate error or another appropriate localization measure.

Even with this limitation, the experiment gave me a practical intermediate stage between detecting the face and predicting its detailed landmarks.

### 4. What's Next?

At this point, the pipeline had progressed from detecting a cat's face to predicting the approximate centers of its five facial regions.

The next step was to use these regions to predict the detailed facial landmarks. Instead of relying only on the five center points, I trained region-specific landmark models to estimate the points associated with each eye, ear, and nose area.

These detailed landmarks would provide the geometric information needed for the later stages of MoodMeow, where facial characteristics could be extracted and used to infer a cat's emotional state.

The next article explores how I approached facial landmark detection and how the five region-specific models contributed to the overall pipeline.
