---
title: From Computer Vision Models to a Mobile App - Building MoodMeow
parent: MoodMeow - Cat Emotion Recognition
nav_order: 8
layout: default
---

## Introduction

Developing computer vision models was only part of my cat emotion recognition project. To make the results accessible beyond a research notebook, I also built **MoodMeow**, an Android application that allows users to upload a cat photo and receive a predicted mood.

The application brings together the different components developed throughout the project, from cat detection and facial landmark prediction to rule-based mood inference. I also designed features that allow users to keep track of their cats' moods over time.

This stage of the project taught me that building a machine learning application involves more than developing a model. The predictions must be integrated into a working application, the user experience must be understandable, and the limitations of the underlying approach must be communicated clearly.

## 1. Designing the user experience

I wanted MoodMeow to feel like a small, playful application rather than a conventional machine learning demo. I therefore chose a retro, pixel-art-inspired visual style, with pixel-style typography, pastel colours, and cat-themed labels and interactions.

The application is organised around a simple idea: users provide a cat photo, receive a mood prediction, and can explore additional features related to their cat's daily mood.

The main experience includes:

- **Photo input:** Users can provide a cat photo for analysis.
- **Mood prediction:** The application displays the predicted mood after processing the image.
- **Daily mood tracking:** Users can record and revisit mood entries through a calendar.
- **Discover and Tips sections:** These provide additional content beyond the initial prediction.
- **Mood Dive:** Users can provide additional information about their cat's posture and tail, allowing the initial prediction to be refined.

The interface was designed to make the results easy to explore without requiring users to understand the underlying computer vision models.

> **Add screenshots here:** The home screen, a prediction result, the mood calendar, and the Mood Dive screen would make this section much stronger. Use screenshots from the actual published application rather than recreating the interface for the article.

## 2. Integrating the computer vision pipeline

The main technical challenge was connecting the individual models into a coherent inference workflow.

The models had been developed as separate components, each with a specific responsibility. In the application, these components needed to work together in the correct order so that the output of one stage could be used by the next.

The high-level workflow is:

```text
Cat Photo
    |
    v
Cat Presence Detection
    |
    v
Frontal Face Check
    |
    v
Cat Face Localization
    |
    v
Facial Region Detection
    |
    v
Facial Landmark Prediction
    |
    v
Geometric Feature Extraction
    |
    v
Rule-Based Mood Inference
    |
    v
Mood Result
```

### Step 1: Checking the input image

The pipeline begins by checking whether the image contains a cat. I used YOLOv7-tiny for cat presence detection and a Haar Cascade classifier as an additional check for a frontal cat face.

These components serve different purposes. YOLOv7-tiny checks for a cat, while the Haar Cascade is used primarily to assess whether a frontal face is visible. Neither is responsible for producing the final facial crop.

### Step 2: Localizing the face and facial regions

After the initial checks, the face localization model predicts a bounding box around the cat's face. A separate region detection model then predicts the approximate centres of five facial regions:

- Left eye
- Right eye
- Left ear
- Right ear
- Nose and mouth area

Before extracting the individual regions, I aligned the face so that the two eyes were approximately on the same horizontal level. This helped establish a more consistent orientation for the subsequent processing stages.

### Step 3: Predicting facial landmarks

Each facial region is processed by its corresponding landmark detection model. Together, these models predict 48 facial landmarks.

The landmark coordinates are then used to calculate geometric features, including eye openness, estimated pupil size relative to the eye, mouth openness, and ear orientation.

### Step 4: Inferring the mood

Rather than feeding the landmarks into a separately trained emotion classification model, I implemented a set of rules that interprets the extracted geometric features.

The rules map combinations of eye, pupil, mouth, and ear states to one of five mood labels:

- Happy
- Playful
- Curious
- Angry
- Afraid

This distinction is important: the final mood prediction comes from handcrafted rules applied to the landmark-derived features, rather than from a model trained directly on labelled cat emotions.

The complete methodology is described in the preceding articles in the [Cat Emotion Recognition Journey](./).

## 3. Running inference in the application

The computer vision components were integrated into the Flutter application using ONNX Runtime. This allowed the application to execute the exported models as part of its inference workflow.

The application therefore acts as the interface and orchestrator for the pipeline: it receives the image, coordinates the processing stages, and presents the resulting mood to the user.

Integrating the models required attention to more than their individual predictions. Their inputs and outputs also had to be compatible across stages. For example, face localization must provide a usable crop for region detection, and region detection must provide locations that can be used to extract the appropriate facial regions for landmark prediction.

An error or unsuitable prediction early in the pipeline can affect every subsequent stage. A technically valid output from an individual model does not necessarily guarantee a meaningful final mood prediction.

This is one of the key differences between experimenting with models independently and integrating them into an end-to-end application.

## 4. Turning predictions into an ongoing experience

I wanted MoodMeow to offer more than a one-time prediction. I therefore added features that allow users to revisit their cats' moods and interact with the application beyond the initial image analysis.

### Daily mood tracking

The calendar allows users to review mood entries by date. I used local persistence to maintain the daily records between application sessions.

This feature turns an individual prediction into something users can revisit over time. However, the records should be understood as a history of the application's predictions or user entries, not as an objective measurement of a cat's emotional state.

### Mood Dive

Mood Dive extends the initial facial analysis by incorporating additional information about the cat's posture and tail.

The application uses this information together with the initial prediction to refine the result. This is a rule-based refinement step rather than an independently validated emotion model.

The feature reflects a broader design consideration: facial appearance alone may not provide enough information to interpret an animal's behaviour. Additional contextual information can be useful, although it does not eliminate the uncertainty of the prediction.

### Discover and Tips

I also included Discover and Tips sections to make the application more engaging and provide content beyond the prediction screen.

Together, these features help connect the machine learning functionality to a more complete user experience.

## 5. Handling limitations and uncertainty

One of the most important lessons from this project is that a working application does not automatically imply that its predictions are scientifically reliable.

There are several limitations to MoodMeow's approach.

**The underlying dataset was not labelled with emotions.** CatFLW provides facial images, bounding boxes, and annotated landmarks, but it does not provide the emotion labels needed to train a supervised emotion classifier directly. The final mood categories therefore rely on my interpretation of facial geometry and the rules implemented in the application.

**The geometric features are heuristic.** Thresholds used to classify eye openness, pupil size, mouth openness, and ear orientation are implementation choices, not universal biological thresholds. The resulting states can also be affected by lighting, image quality, pose, and the coordinate system used for the landmarks.

**Errors can propagate through the pipeline.** If the face or a facial region is incorrectly localized, the landmark model may receive an unsuitable crop. That can distort the geometric features and ultimately affect the predicted mood.

**The rule-based inference does not cover every possible combination.** Some combinations of facial states do not satisfy any of the rules that assign a mood. The application should not be interpreted as having a validated prediction for every possible input.

For these reasons, MoodMeow is an entertainment application rather than a diagnostic tool or a scientifically validated measure of feline emotion. Its results should not be used to assess a cat's health or welfare.

## 6. What I learned from building MoodMeow

Building MoodMeow helped me understand the broader work involved in turning machine learning experiments into an application.

First, **individual model performance is only one part of the system**. A complete workflow must connect each model's inputs and outputs correctly and handle the possibility of errors at different stages.

Second, **the user experience matters**. A prediction is more useful when it is presented in an understandable interface and supported by features that give users a reason to return to the application.

Third, **technical limitations need to remain visible after deployment**. A polished interface can make a prediction appear more authoritative than it really is. Clearly communicating uncertainty is therefore part of responsible application design.

Finally, this project gave me practical experience connecting computer vision, rule-based logic, local data persistence, and mobile application development in one end-to-end product.

## Conclusion

MoodMeow brought together the different stages of my cat emotion recognition project, from dataset selection and model development to facial feature extraction, mood inference, and mobile application integration.

Although the approach has important limitations and should not be considered a validated emotion recognition system, completing the application allowed me to explore the challenges of making a machine learning workflow usable outside a research environment.

The project also reinforced an important lesson: building an end-to-end machine learning application is not only about producing a prediction. It is about connecting the technical components, designing a coherent user experience, and being transparent about what the system can and cannot tell us.

**Project:** [MoodMeow on GitHub](https://github.com/seyong2/moodmeow)

**Note:** The source repository is private. The link above is intended for authorised readers; the application source code is not publicly accessible through this article.
