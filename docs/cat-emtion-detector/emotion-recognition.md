---
title: Recognizing Cat Emotion
parent: Cat Emotion Recognition Journey
nav_order: 8
layout: default
---

In this final post on my Cat Emotion Recognition Journey, I want to briefly discuss how I determined cat emotions using landmark coordinates. So far, we’ve explored three types of model architectures that allow us to predict 48 cat facial landmarks. Now, the time has come—we may finally be able to **interpret a cat’s emotions based on the position of the landmarks predicted by the ensemble landmark detection model**.

The first topic in this journey was understanding what kind of emotions cats can experience. I referred to an ethogram by Nicholson, S.L. and O’Carroll, R.Á to identify feline emotions and found that five core emotions can be recognized in animals:

- Fear
- Anger
- Joy
- Contentment
- Interest

We also saw that these emotions can be identified through cats' body language. **For emotion classification, I focused on key facial signals—specifically, the shape and position of the eyes, ears, and nose area**.

### Example: Detecting Pupil Dilation

Rather than going into full detail on classifying each emotion, I’d like to demonstrate a simple example: **how to determine whether a cat’s pupils are dilated**—a crucial factor in cat emotion recognition.

Take this example image from the previous post:

<p align='center'> 
  <img src="https://github.com/user-attachments/assets/cc8052bf-13ab-4a68-9fe3-4d9f895b3613" width=300> 
</p>

To check if the pupil is dilated, we can look at how much of the eye area the pupil occupies. **If the ratio of pupil area to total eye area is below a certain threshold, we might classify it as miotic (constricted). Otherwise, we can consider it dilated**.

The same logic can be applied to other facial features, such as:

- Determining whether the mouth is open or closed
- Checking if the ears are facing forward or swiveled sideways
- By analyzing these facial features, we can infer a cat’s emotional state.

### Final Thoughts & Next Steps

Looking back, I initially thought I wouldn’t be able to complete this project—it seemed too challenging at first. However, as I kept moving forward, I learned a lot, which motivated me to keep pushing through.

One limitation of this project is that the **CatFLW dataset lacks landmark data for a cat’s tail and other body parts**, which could provide additional emotional cues. Despite this, I believe CatFLW was an excellent starting point, and it has inspired me to explore the possibility of creating a more diverse dataset in the future.

Ultimately, **my end goal is to build a mobile application that cat owners can use to understand their cat’s emotions in real time**. While I won’t be discussing that project here, I’ll definitely share a link to the app once it’s published so you can try it out!

Thank you all so much for accompanying me on this journey!
