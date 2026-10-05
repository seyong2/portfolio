---
title: CatFLW Dataset
parent: MoodMeow - Cat Emotion Recognition
nav_order: 2
layout: default
---

After defining the emotional states I wanted to recognize and identifying the facial characteristics associated with them, the next challenge was finding a dataset that would allow me to measure those characteristics.

I was looking for a dataset that could provide:

* A sufficient number of cat face images
* **Face bounding boxes** to locate the facial area
* **Facial landmarks** to identify specific facial structures
* Enough variation in the images to make the models useful beyond a single controlled environment

Finding a dataset with all of these characteristics turned out to be much easier than I expected.

### What Data Did I Need?

The feline emotions ethogram gave me a starting point: different emotional states are associated with observable changes in features such as the eyes, ears, mouth, and overall body posture.

Since I wanted to build a computer vision system, I needed to turn these observations into something measurable.

For example:

* Are the cat's eyes open or narrowed?
* Are the ears pointing forward or flattened?
* What is the shape and position of the mouth?
* Where are the relevant facial features located?

To answer questions like these programmatically, I needed more than just images of cats. I needed annotated facial landmarks that could tell my models where these features were located.

This led me to the **CatFLW dataset**.

### Why CatFLW?

CatFLW is a dataset specifically designed for cat facial landmark detection. It contains **2,079 cat face images**, each annotated with **48 facial landmarks** and a bounding box around the face.

This made it particularly suitable for my project because the annotations provide information about the exact location of facial structures rather than simply indicating whether an image contains a cat.

The dataset was developed by researchers at the **University of Haifa in Israel** to study cat facial characteristics, with applications including the analysis of pain and emotional states.

The images also contain considerable variation in the cats and their environments, which makes the dataset more useful for developing models intended to work with real-world images.

Most importantly, the dataset's annotations gave me the foundation I needed to move from a qualitative description of cat emotions to measurable facial characteristics.

### What's Inside the Dataset?

Each image in CatFLW contains two important types of annotations:

#### Face bounding box

A bounding box identifies the location of the cat's face within the image.
This provides a starting point for isolating the relevant area before performing more detailed facial analysis.

#### 48 facial landmarks

Each cat face is annotated with **48 landmarks** representing important points around the eyes, ears, nose, and mouth.

These landmarks allow the shape and relative position of different facial structures to be represented numerically.

For example, instead of simply saying that a cat has "narrow eyes," I can use the positions of the corresponding landmarks to calculate geometric measurements that describe the eye shape.

This is the key connection between the dataset and the emotion framework I explored in the previous article:

> **Emotional characteristics → observable facial features → facial landmarks → measurable features**

The landmarks therefore became the foundation of the computer vision pipeline I would eventually use in MoodMeow.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d0b1bf49-7916-4b63-807b-ce98ca371f0a">
</p>

### How Were the Landmarks Annotated?

One aspect of CatFLW that I found particularly interesting was how the annotations were created.

The researchers used an **AI-assisted, human-in-the-loop annotation process**. Rather than manually annotating every image entirely from scratch, model predictions were used to assist human annotators, who then reviewed and refined the results.

This approach helped reduce the amount of manual annotation work while maintaining the quality of the landmark annotations.

The researchers reported that this process reduced annotation time by approximately **35%** compared with their previous annotation approach.

The annotations were also designed around feline facial anatomy and the structure of feline facial musculature, making them particularly relevant to facial expression analysis and frameworks such as **CatFACS**.

### The Challenge: No Emotion Labels

At first, CatFLW seemed almost perfect for my project.

There was just one important problem.

**The dataset does not contain explicit emotion labels.**

In other words, the dataset tells me:

> "Here are the locations of the cat's facial landmarks."

But it does not tell me:

> "This cat is happy."

This meant I could not simply take the CatFLW images and train a conventional supervised five-class emotion classifier.

And this turned out to be an important part of how the project evolved.

The goal was no longer simply:

```text
Cat image → Emotion
```

Instead, I needed to break the problem into several stages:

```text
Cat image
    ↓
Locate the cat
    ↓
Locate the face
    ↓
Detect facial landmarks
    ↓
Extract measurable facial characteristics
    ↓
Relate those characteristics to the feline emotion framework
    ↓
Infer emotional state
```

This approach also made the system more interpretable. Rather than relying entirely on a black-box classifier, I could understand which observable characteristics were contributing to the final prediction.

### What This Meant for My Project

CatFLW gave me the data I needed to locate the facial structures I was interested in, but it did not directly solve the emotion-recognition problem.

That meant I needed to build the solution in stages.

First, I needed models capable of detecting the relevant facial landmarks. Then, I could use those landmarks to calculate geometric features describing the eyes, ears, and mouth. Finally, I could relate those measurable characteristics to the emotional states described in the feline emotions ethogram.

This became the foundation of the technical approach behind MoodMeow:

**CatFLW → Landmark Detection → Feature Extraction → Emotion Inference**

The next question was therefore no longer *"Which dataset should I use?"*

It was:

**How can I reliably detect these facial features from a real-world cat image?**

That led me to the next stage of the project: **face detection and localization**.

---
#### Resources

* [CatFLW Dataset](https://www.kaggle.com/datasets/limingsun/catflw-dataset)
* [CatFLW: Cat Facial Landmark Dataset](https://github.com/liming-sun/CatFLW)
