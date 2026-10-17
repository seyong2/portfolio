---
title: Inferring Cat Emotions from Facial Landmarks
parent: MoodMeow - Cat Emotion Recognition
nav_order: 7
layout: default
---

After detecting the cat's face, identifying five facial regions, and predicting 48 facial landmarks, the next challenge was to turn the landmark coordinates into an interpretable mood prediction.

Instead of training a separate deep learning model to classify emotions directly from images, I took a rule-based approach. I extracted geometric features from the predicted landmarks and used predefined thresholds and conditional rules to infer a possible mood.

This approach allowed me to connect the computer vision pipeline to explicit, interpretable logic. Rather than producing a prediction without an obvious explanation, the application could evaluate characteristics such as eye openness, pupil appearance, ear orientation, and mouth openness before assigning a mood.

The resulting system was designed for MoodMeow, an entertainment-oriented Android application. Its predictions are heuristic interpretations of visible facial features, not definitive measurements of a cat's internal emotional state.

## 1. From Facial landmarks to Geometric Features

The landmark detection stage produced 48 facial landmarks across five regions: the left eye, right eye, left ear, right ear, and nose/mouth area.

These landmarks provided coordinates that could be used to calculate geometric features. I focused on four types of facial characteristics:

* Eye openness
* Pupil appearance relative to the eye
* Ear orientation
* Mouth openness

Each characteristic was derived from selected landmark coordinates. The resulting feature categories were then passed to the rule-based inference logic.

## 2. Extracting Facial Features

### 2.1. Eye Openness

I estimated eye openness using an eye aspect ratio, calculated by dividing a vertical distance between eye landmarks by a horizontal distance. 

The ratio provides a simple geometric indicator of how open the eye appears. A relatively small vertical distance compared with the eye's width produces a lower ratio, while a larger vertical distance produces a higher ratio.

I used the following thresholds to categorize each eye:

| Eye aspect ratio | Category |
|---|---|
| Below 0.1 | Tightly closed |
| From 0.1 to 0.3, inclusive | Half open |
| Above 0.3 | Wide open |

I applied this calculation separately to the left and right eyes.

These thresholds were heuristic decision boundaries used in my implementation. They were not learned by a model or established as universal biological thresholds for feline eye states.

### 2.2. Pupil Appearance

For eyes classified as wide open, I also estimated the pupil's size relative to the eye.

I approximated both the pupil and the eye as ellipses. the area of an ellipse is:

$$A=\pi ab$$

where $$a$$ and $$b$$ are the semi-major and semi-minor axes.

I estimated these axes from selected landmark distances. For example, for the left eye, the vertical distance between two landmarks provided an estimate of the pupil's semi-major axis, while a horizontal distance provided an estimate of its semi-minor axis. I used corresponding measurements for the eye itself.

I then calculated the pupil-to-eye area ratio:

$$\text{Dilation Ratio}=\frac{\text{Estimated Pupil Area}}{\text{Estimated Eye Area}}$$

This ratio provided a relative measure of pupil size within the eye, rather than relying on the pupil's absolute pixel dimensions.

I categorized the ratio as follows:

| Dilation ratio | Category |
|---|---|
| At most 0.1 | Miotic |
| Above 0.1 and at most 0.3 | Dilated |
| Above 0.3 | Mydriatic |

In my implementation, pupil appearance was evaluated separately for each eye. The pupil-area calculation was performed by the pupil-state functions, independently of the eye-openness classification.

**An important implementation detail**: the mood-inference logic used the miotic category explicitly. Although the feature-extraction stage also distinguished dilated and mydriatic pupils, the final rules did not use those two categories to distinguish between moods.

Pupil size can vary with lighting, arousal, and other physiological factors. Therefore, this ratio should be understood as a visual feature used by the application's heuristic logic, rather than a direct measurement of emotional state.

### 2.3. Ear Orientation

I used the landmarks associated with each ear to estimate its orientation.

For each ear, I calculated two ratios from selected horizontal and vertical landmark coordinates. These ratios were compared with a threshold of 0.8 to classify the ear into one of three categories:

* Forward facing
* Flat
* Swiveled sideways

The rules first evaluated whether the ear appeared swiveled sideways. If that condition was met, a second ratio was used to distinguish a flat ear from one swiveled sideways. Otherwise, the ear was classified as forward-facing.

I applied separate functions to the left and right ears because their landmark arrangements differed.

These calculations were simple geometric heuristics based on the coordinates predicted by the landmark models. They were not a trained ear-pose classifier, and the resulting categories should not be interpreted as universally reliable measurements of ear orientation.

### 2.4. Mouth Openness

To estimate mouth openness, I used two landmarks from the nose/mouth region. Specifically, I selected one landmark representing the upper part of the mouth and another representing the lower part.

I calculated a ratio from their vertical coordinates:

The ratio was then mapped to one of three categories:

| Mouth ratio | Category |
|---|---|
| Below 1.2 | Closed |
| From 1.2 to below 1.5 | Half open |
| At least 1.5 | Wide open |

These thresholds allowed the application to represent mouth appearance using a small number of discrete states instead of working directly with all the original landmark coordinates.

As with the other features, this was a heuristic calculation. Because it uses ratios of raw coordinates rather than distances between points, its behavior can depend on the coordinate system, crop position, and image orientation. It should not be treated as a general-purpose, translation-invariant measurement of mouth openness.

## 3. Rule-Based Mood Inference

After extracting the features, I combined them using a set of conditional rules to infer one of five moods:

* Happy
* Playful
* Curious
* Angry
* Afraid

The system did not use a learned classifier to map features to these labels. Instead, the rules explicitly combined the eye, pupil, ear, and mouth categories.

### Step 1: Check the ears

The first decision was whether both ears were classified as forward-facing.

If both ears were forward-facing, the application evaluated the eye and pupil features. Otherwise, it followed a different set of rules based on the eye and mouth states.

### Step 2: When both ears are forward-facing

The application first checked whether either pupil was classified as miotic, or whether both eyes were not wide open.

If either condition was satisfied, the predicted mood was **Happy**.

Otherwise, if neither pupil was miotic, the mouth state determined the next decision:

* If the mouth was not closed, the predicted mood was **Playful**.
* If the mouth was closed, the predicted mood was **Curious**.

This branch used the eye and pupil features together with mouth openness to distinguish between the three possible outcomes.

### Step 3: When both ears are not forward-facing

If both ears were not classified as forward-facing, the application checked whether both eyes were wide open or both were tightly closed.

If either of these conditions was satisfied, the mouth state determined the predicted mood:

* If the mouth was not closed, the predicted mood was **Angry**.
* If the mouth was closed, the predicted mood was **Afraid**.

The rules therefore combined multiple facial characteristics instead of relying on a single feature to determine the output.

### Summary of the decision rules

| Ear condition | Eye and pupil condition | Mouth condition | Predicted mood |
|---|---|---|---|
| Both ears forward-facing | Either pupil is miotic, or neither eye is wide open | Not evaluated | Happy |
| Both ears forward-facing | Neither pupil is miotic, and both eyes are wide open | Not closed | Playful |
| Both ears forward-facing | Neither pupil is miotic, and both eyes are wide open | Closed | Curious |
| Both ears not forward-facing | Both eyes are wide open or both are tightly closed | Not closed | Angry |
| Both ears not forward-facing | Both eyes are wide open or both are tightly closed | Closed | Afraid |

The table summarizes the conditions that explicitly assign a mood in the implementation. It does not cover every theoretically possible combination of features.

For example, when both ears are not forward-facing but the eyes do not satisfy either of the two specified conditions, the displayed decision logic does not assign a mood. Similarly, the forward-facing branch relies on the expected pupil-state outputs when evaluating its nested conditions. These are implementation-specific constraints rather than a complete classification of every possible input.

## 4. Interpreting the Results

The main advantage of this approach was its interpretability. Each feature had a defined calculation, each calculation produced a discrete category, and the final mood prediction followed explicit conditional rules.

This also made the system relatively straightforward to inspect and modify. For example, I could adjust a feature threshold or change a conditional rule without retraining a deep learning model.

However, interpretability does not automatically imply accuracy.

The quality of the final prediction depended on several stages working together:

1. The face and facial regions had to be localized sufficiently well.
2. The landmark models had to predict useful facial coordinates.
3. The geometric calculations had to represent the intended visual characteristics.
4. The thresholds and conditional rules had to produce reasonable interpretations of those characteristics.

An error in an early stage could affect the extracted features and, consequently, the final mood prediction.

Furthermore, the rules were handcrafted rather than trained and validated against a labeled dataset of cat emotions. The five output categories therefore represented the application's intended interpretations, not experimentally established predictions of a cat's internal emotional state.

## 5. Limitations and Potential Improvements

Looking back at this approach, there are several limitations worth acknowledging.

### Sensitivity to image conditions

The extracted features depended on the accuracy of the predicted landmarks. Image quality, viewing angle, face orientation, and occlusion could affect landmark localization and, in turn, the calculated features.

The pupil-to-eye ratio also depended on visible pupil boundaries and could vary with lighting. A large pupil, for example, cannot be attributed to a particular emotion from a photograph alone.

### Handcrafted thresholds

The thresholds were predefined rather than learned from labeled examples. Although this made the logic easy to interpret, it did not establish that the selected thresholds were optimal or generalized well across different cats and image conditions.

A possible improvement would be to evaluate alternative thresholds using a carefully labeled dataset, provided that reliable ground-truth labels and an appropriate evaluation methodology were available.

### Coordinate-based ratios

Some of the ear and mouth calculations used ratios of raw coordinates. Such ratios can depend on the coordinate origin and image orientation, rather than only on the relative geometry of the facial features.

A more robust implementation could use distances and angles between landmarks, or normalized geometric features designed to be less sensitive to image position and scale.

### Evaluation of emotion predictions

Accurate landmark localization does not necessarily imply accurate emotion inference. The rule-based system would need to be evaluated separately against suitable labeled observations before making claims about its emotion-recognition performance.

A more rigorous evaluation could also investigate whether the same rules work across different cats, lighting conditions, and image angles.

## 6. Final Thoughts

Building MoodMeow allowed me to explore how computer vision models and handcrafted logic could work together in a practical application.

The project progressed from detecting cat faces to predicting facial landmarks, extracting geometric features, and combining those features through interpretable rules. Rather than treating the output of a neural network as the final answer, I used its predictions as the basis for a second layer of reasoning.

This approach gave me practical experience with computer vision, regression-based landmark prediction, feature engineering, and rule-based decision systems. It also highlighted an important distinction in applied machine learning: a pipeline can produce a working prediction without that prediction necessarily being scientifically validated.

MoodMeow remains an entertainment-oriented application that offers a playful interpretation of a cat's visible facial characteristics. Its predictions should not be treated as a definitive assessment of feline emotions.
