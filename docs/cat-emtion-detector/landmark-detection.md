---
title: Facial Landmark Detection
parent: Cat Emotion Recognition Journey
nav_order: 7
layout: default
---

In the previous post on [Identifying 5 Cat Face Regions](https://seyong2.github.io/portfolio/docs/cat-emtion-detector/region-detection.html), I explained how I trained a model to identify the center points of five primary regions in a cat's face. 

The final step- as suggested in the paper that I've been following to replicate the architecture by Martvel, G. et al.- is to build **five separate models**, each responsible for predicting landmark coordinates within its corresponding facial region. Once the models generate their predictions, we combine the outputs to produce the final set of 48 facial landmarks. If you need a refresher on the model architecture, check out my previous post on [How to Detect Cat Facial Landmarks?](https://seyong2.github.io/portfolio/docs/cat-emtion-detector/prep.html).

<p align="center">
  <img src="https://github.com/user-attachments/assets/d63e47e4-ed06-4a1b-8be5-0575f9f92e8c" title="ensemble-detection">
</p>

To train the models, I used the **EfficientNetV2 architecture**, modifying the top layers to **accommodate the desired output sizes**. Additionally, I applied **augmentation techniques** to the training dataset to enhance **generalization** and improve performance on unseen images.

The test images below display the **predictions (in red) from the five models**. As seen, the predicted landmarks are **closely aligned with the ground truth**, demonstrating that the models perform accurately and reliably.

<p align="center">
  <img src="https://github.com/user-attachments/assets/cc8052bf-13ab-4a68-9fe3-4d9f895b3613" width=200>
  <img src="https://github.com/user-attachments/assets/bbf20aaa-f376-4e58-a622-e5b45b671462" width=200>
  <img src="https://github.com/user-attachments/assets/207f3071-4881-424e-9655-c05e2498362e" width=200>
  <img src="https://github.com/user-attachments/assets/add2ee33-6cec-4c30-bab7-35250c2ade0b" width=200>
  <img src="https://github.com/user-attachments/assets/55adf724-8b44-4e93-8f01-f1e5df2045a3" width=200>
</p>

With this, the **landmark detection pipeline is complete**, marking a major milestone in this journey. Now, it's time to bring everything together. In the next and final post, I'll reflect on the entire process, discuss key takeaways, and explore the final step—**using these facial landmarks to analyze cat emotions**. Stay tuned!
