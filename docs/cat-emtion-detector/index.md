---
title: MoodMeow - Cat Emotion Recognition
nav_order: 3
layout: default
---

An end-to-end computer vision project for recognizing feline emotional states from face images, ultimately deployed as the Android application [**MoodMeow**](https://play.google.com/store/apps/details?id=com.meowlytics.moodmeow&hl=en).

## Project Overview

MoodMeow uses computer vision and facial landmark analysis to estimate a cat's emotional state from a photo. The project combines object detection, facial landmark detection, feature engineering, and machine learning into an end-to-end pipeline that can run on a mobile device.

The final application predicts five primary moods:

**Happy**, **Afraid**, **Angry**, **Curious**, and **Playful**.

### The ML Pipeline

<p align="center">
  <img alt="MoodMeow_Pipeline" src="https://github.com/user-attachments/assets/1a79b0ac-d73d-469d-ae63-25f27f311d91">
</p>

#### Dataset

The computer vision pipeline was developed using **CatFLW**, a dataset containing 2,079 cat-face images annotated with bounding boxes and 48 facial landmarks. The dataset provides the visual information needed to detect and analyze facial features, while the emotion-recognition stages builds on these features to infer a cat's emotional state.

#### Technologies

##### Machine Learning

- Python
- PyTorch
- EfficientNetV2
- YOLO
- Computer vision
- Facial landmark detection

##### Model Development

- ONNX
- ONNX Runtime
- On-device inference

##### Application

- Flutter
- Android

### What Makes This Project Different

Rather than stopping at training a model, I wanted to take the project through the entire process:

Research &rarr; Data &rarr; Modeling &rarr; Experimentation &rarr; Evaluation &rarr; Development &rarr; Application

This meant dealing not only with model performance, but also with challenges such as limited labeled data, object and landmark detection, model conversion, and running inference efficiently on a mobile device.


### Why I Built This

I decided to dedicate myself to this project, creating a mobile application that helps cat owners understand how their cat is feeling. Long story short, my husband had the opportunity to pursue a PhD in Amsterdam, and since we had always wanted to live abroad, we quit our jobs and moved to the Netherlands to start a new chapter in our lives.

As a foreigner, it took me some time to get a work permit, and naturally, I got bored. Wondering how to keep myself entertained in the meantime, I thought, “Why not work on a cool data science project that could also be a plus when job hunting?” I could have joined a Kaggle competition, but I wanted to try something that truly interested me.

Since I love cats, I figured, “Why not do something cat-related? That way, I get to spend all day looking at cat photos!” And that’s how it all started. I came across a fascinating dataset of cat faces with facial landmarks and thought it would be the perfect foundation for a project, which is developing an algorithm that detects facial landmarks on a cat’s face and determines how the cat is feeling.

What started as a personal data science project eventually grew into a complete computer vision pipeline and a mobile application. The sections below walk through how I approached each stage, from understanding feline emotions and preparing the dataset to developing, evaluating, and deploying the models.
