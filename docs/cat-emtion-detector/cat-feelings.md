---
title: What Emotions Do Cats Feel?
parent: MoodMeow - Cat Emotion Recognition
nav_order: 1
layout: default
---

Before building any models, I first needed to understand what visual and behavioral cues could be used to infer a cat's emotional state.

At first, I thought this would be straightforward. However, I quickly realized that there's no single, universally accepted framework for describing feline emotions. During my research, I came across a study by Nicholson and O'Carroll that proposed an ethogram for describing five recognizable feline emotional states: **fear, anger/rage, joy/play, contentment, and interest**.

An ethogram is a systematic description of observable behaviors and physical cues associated with different states. For my project, this provided a useful starting point: instead of trying to predict a cat's emotional state without any domain knowledge, I could first identify the visual characteristics that might provide useful signals.

### Feline Emotional States

The following table summarizes some of the physical and behavioral characteristics described in the ethogram:

| Emotional State | Eye | Ears | Mouth / Face | Other Behavioral Cues |
| --- | --- | --- | --- | --- |
| **Fear** | Wide eyes, dilated pupils | Flattened sideways of backward | - | Body lowered, tail tucked |
| **Anger/Rage** | Dilated pupils | Turned sideways or backward | Open mouth, exposed teeth | Rigid posture, tail lashing |
| **Joy/Play** | Round or dilated pupils | Upright and forward-facing | Open "play face" | Playful posture |
| **Contentment** | Miotic pupils | Forward-facing | Relaxed facial expression | Relaxed body posture |
| **Interest** | Round or dilated pupils | Directed toward the stimulus | - | Alert and attentive posture | 

<p align="center">
  <img src="https://github.com/user-attachments/assets/7701e8af-2bf2-4b75-b0e3-299ff16c839e" width="400" height=800 title="feline-emotions-images">
</p>

These characteristics are not intended to be interpreted as definitive indicators of an emotion on their own. A cat's facial appearance can be influenced by other factors as well. For example, pupil size can be affected by both emotional arousal and ambient light conditions.

This is particularly important when trying to translate behavioral research into a computer vision problem. A single visual feature may not uniquely correspond to one emotional state. Instead, multiple cues need to be considered together.

### From Research to Measurable Features

The ethogram gave me a set of observable characteristics to look for, but the next question was:

**How can these characteristics be measured from an image?**

For this project, I focused primarily on facial cues that could be extracted using facial landmarks:

- Eyes: eye shape and degree of eye constriction
- Ears: ear orientation
- Mouth: mouth shape and opening

Rather than feeding the entire image directly into a black-box emotion classifier, I wanted to build a pipeline that could first identify these facial structures and then derive interpretable measurements from them.

This led to the next stage of the project: finding a suitable dataset containing cat facial landmarks and developing a model capable of detecting them.

### A Note on the Research

The ethogram served as a domain-knowledge foundation for MoodMeow rather than as a ground-truth emotion dataset. The study provides a framework for describing observable characteristics associated with different feline emotional states, but these characteristics should not be interpreted as a definitive measurement of a cat's internal emotional state.

This distinction became important throughout the project. My goal was not to claim that a particular landmark or facial feature directly reveals what a cat is feeling. Instead, I used combinations of measurable facial characteristics as signals for emotion inference.

With this foundation in place, I could move on to the computer vision problem: how to locate and measure these facial features automatically.

---
#### Resources
- Nicholson, S.L., O’Carroll, R.Á. Development of an ethogram/guide for identifying feline emotions: a new approach to feline interactions and welfare assessment in practice. Ir Vet J 74, 8 (2021). https://doi.org/10.1186/s13620-021-00189-z
