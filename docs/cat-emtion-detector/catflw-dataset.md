---
title: CatFLW Dataset
parent: Cat Emotion Recognition Journey
nav_order: 2
layout: default
---

In the previous post, we explored the five primary emotions cats can experience are. Now, it's time to see whether I'm lucky enough to obtain relevant data to apply the feline emotions ethogram. Unlike the first task (define different cat emotions), this was not as difficult as I had expected. 

Some researchers at the University of Harifa in Israel published a high-quality dataset containing 2,079 cat face images with 48 annotated facial landmarks. Their goal was to analyze cat facial expressions, particularly for pain and emotion recognition-similar to what I'm working on now. The annotation of facial landmarks was based on feline facial musculature and their relevance to cat-specific Facial Action Units (CatFACS), ensuring a biologically meaningful representation.

According to their paper [CatFLW: Cat Facial Landmarks in the Wild Dataset](https://arxiv.org/pdf/2305.04232), the dataset is the largest of its kind, surpassing existing animal landmark datasets. The images were collected in diverse environments to ensure diversity. Besides 48 facial landmarks, each image also contains bounding box coordinates that encompass only the cat's face.

Given that each image includes 48 landmarks plus a bounding box, the researchers used an AI-assisted, human-in-the-loop annotation method to speed up the labeling process and reduce manual work. This semi-supervised learning approach iteratively trained a model on annotated data, allowing it to predict landmarks on new images, which human annotators then refined. This method reduced annotation time by 35%, making large-scale labeling more feasible.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d0b1bf49-7916-4b63-807b-ce98ca371f0a">
</p>

Even though this dataset only includes facial landmark data, it serves as a strong starting point. By analyzing how these landmarks are positioned relative to one another, we may still be able to infer a cat's emotional state. However, since the dataset does not contain explicit emotion labels, additional modeling or validation might be needed to fully map landmarks to specific emotional states.

---
#### Resources
- CatFLW Dataset. https://www.kaggle.com/datasets/georgemartvel/catflw/data
