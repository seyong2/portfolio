---
title: MLOps
parent: Data Science
nav_order: 1
layout: default
---

As part of building my application to identify cats' emotions, I realized that deploying the model is just as important as training it. That's where machine learning operations (MLOps) come in. MLOps covers the entire lifecycle of a machine learning project— from data definition and model training to deployment and monitoring. In this section, I'll be sharing key MLOps concepts I'm learning from **DeepLearning.AI's MLMachine Learning in Production** course, which are essential for deploying models effectively. 

The course starts by outlining the machine learning project lifecycle, from scoping the project to training models and selecting deployment strategies.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8da5a6b8-5d22-4730-9ad6-e754b7cb93d3">
</p>

### 1. Scoping 

We must first define the project we're working on. In my case, I'm currently developing a computer vision model to identify cats' emotions through facial landmarks. Defining the project also means establishing key evaluation metrics. For my application, the focus is three things: accuracy (how precise the outputs are), latency (how fast the results are generated), and throughput (how many queries it can handle per second). While I haven’t fully scoped out the resources yet, this will also be a key part of the process as the project evolves.

### 2. Data

In the second stage, we define the data, establish a baseline, and organize the necessary data for the project. In my [Cat Emotion Recognition Journey](https://seyong2.github.io/portfolio/docs/cat-emtion-detector/), I explained the data I would use: annotated cat images with bounding boxes and 48 facial landmarks. 

A major challenge in this stage is ensuring data inconsistency, as inconsistently labeled data can negatively impact model performance. While I haven't encountered issues so far, it’s important to remain vigilant for potential inconsistencies, as even widely used academic datasets can contain errors. High-quality data often contributes more to model performance than adjustments to the model itself.

### 3. Modeling

The modeling phase involves selecting and training the model, as well as performing error analysis. The three key inputs for model training are:

- Code (the neural network model architecture)
- Hyperparameters
- Data

Although we tend to focus on optimizing the model and hyperparameters, it can sometimes be more effective to prioritize refining the data. Error analysis helps identify where the model falls short and whether issues stem from the data or the code.

### 4. Deployment

Once error analysis shows the model is performing well, we move to deployment. For example, I might have software on a mobile phone that captures a cat’s image via the camera. The image would then be sent to a prediction server, likely hosted in the cloud. The server would return the emotion prediction, which is displayed on the phone.

Even after deployment, ongoing monitoring and maintenance are essential. One challenge we may encounter is concept or data drift, which occurs when the data distribution changes. Promptly addressing such drifts ensures the system continues to provide value as intended.

#### Resources
- [Machine Learning in Production by DeepLearning AI](https://www.coursera.org/learn/introduction-to-machine-learning-in-production)
