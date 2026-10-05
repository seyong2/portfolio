---
title: What Emotions Do Cats Feel?
parent: MoodMeow - Cat Emotion Recognition
nav_order: 1
layout: default
---

Before building any models, I first needed to understand what visual and behavioral cues could be used to infer a cat's emotional state.

At first, I thought this would be a simple task—but it wasn't. While researching, I realized that there is no clear consensus on how feline emotions should be defined or identified.

I came across a study by Nicholson, S.L. and O'Carroll, R.Á., which proposed an ethogram for identifying feline emotional states. The researchers proposed five recognizable emotional states in domestic cats: **fear, anger/rage, joy/play, contentment, and interest**.

The authors emphasize that emotions are important indicators of animal well-being and that understanding the relationship between emotional states and behavior can help veterinary professionals assess and interact with cats more effectively.

### The Five Emotional States

The researchers defined the five emotional states as follows:

| Emotion         | Definition                                                                                                                                                                                                                                                      |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fear**        | Negative emotional state caused by immediate perceived danger or the threat of danger and manifested as vigilance and attempts to withdraw or escape.                                                                                                           |
| **Anger/Rage**  | Negative emotional state caused by the frustrated desire to perform actions/achieve goals (including escape or exploration) or by competition for resources. Manifested as aggression or the threat of aggression.                                              |
| **Joy/Play**    | A high-intensity positive emotional state, which may be internally motivated. Manifested as non-functional behaviors involving physical activity (locomotor play), interaction with other individuals (social play), or interaction with objects (object play). |
| **Contentment** | A positive emotional state caused by the fulfilment of the animal's needs and desires and an acceptance of their current state. Manifested as resting, calm, and affiliative behavior.                                                                          |
| **Interest**    | A positive emotional state, caused by the presence of a novel stimulus or stimulus of salience and/or anticipation of engagement. Manifested as attention and orientation to the stimulus and/or seeking behaviors.                                             |

These definitions gave me a useful framework for thinking about the problem from a computer vision perspective. If different emotional states are associated with different behaviors, postures, and facial characteristics, could some of these characteristics be measured from an image?

### The Feline Emotions Ethogram

Building on these definitions, the researchers identified behaviors, postures, and body-language cues associated with each emotional state. They also considered the potential risk of handler injury and welfare issues, making the guide more comprehensive.

To support their ethogram, they included photographs of cats displaying behaviors corresponding to each emotional state. The completed guide was reviewed by two certified clinical animal behaviorists. However, the authors also acknowledged that further field testing would be needed to assess its reliability.

The detailed ethogram is reproduced below:

| Emotion         | Body Language                                                                                                                                            |                                                            |                                                                                                                                            |                                                                                                                  | Actions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Risk of handler injury | Risk of welfare issue |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | --------------------- |
|                 | **Eyes**                                                                                                                                                 | **Ears**                                                   | **Tail**                                                                                                                                   | **Body**                                                                                                         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                        |                       |
| **Fear**        | Wide open eyes with round dilated pupils. Blinking or half blinking. Or eyes tightly shut or avoidance of eye contact. Gaze to left in mild fear states. | Flattened to the side or back. Ear pinnae are not visible. | Tucked under the body or wrapped around it.                                                                                                | Piloerection. Tense muscles. Crouching. Lowered head. Standing with an arched back. Left head turn in mild fear. | Vigilance. Startle. Trembling. Freezing. Hiding. Fleeing/avoidance. Grooming. No maintenance behaviors (eating, drinking, elimination)/sleep.                                                                                                                                                                                                                                                                                                                                                                                             | Moderate               | High                  |
| **Anger/Rage**  | Pupils oblong and dilated. Direct stare.                                                                                                                 | Swivelled sideways. Inner pinnae are visible.              | Lowered and rigid. Held in an inverted L shape. Slapping against the ground. Rapidly moved from side to side (or up and down) (Tail lash). |                                                                                                                  | Exposing teeth. Launching at/chasing individuals. Attacking with paws or mouth. Displace others.                                                                                                                                                                                                                                                                                                                                                                                                                                          | High                   | High                  |
| **Joy/Play**    | Pupils dilated/round due to arousal. Or relaxed/soft.                                                                                                    | Upright and forward facing                                 | Vertical. May take an inverted U shape.                                                                                                    | "Play face" in kittens: a half-open mouth. Arching spine. Body posture varies.                                   | Locomotor play. Climbing. Running. Social play. Approaching cat. Jumping. Patting, pawing playmate. Grabbing playmate with forelimbs. Biting playmate. Rolling/presenting belly. Wrestling playmate. Kicking/raking playmate. Chasing playmate. Side stepping or running away from playmate. Object play. Rearing to reach object. Pawing, batting object. Holding object with paws. Sniffing, licking object. Biting, chewing object. Throwing object. Wrestling with object. Predatory: Stalking, chasing, jumping, pouncing on object. | Moderate               | Moderate              |
| **Contentment** | Pupils are small miotic vertical ovals. Half-open.                                                                                                       | Upright and forward facing.                                | Tail relaxed and still. May be erect and slightly curled.                                                                                  | Sitting. Lying curled up in circular formation.                                                                  | Stretching. Yawning. Grooming self or other (allogrooming). Kneading/treading paws. Friendly greeting (nose touching/sniffing, head butting, rubbing face and body against object/individual-allorubbing). Rolling onto back or from side to side. Nuzzling. Eating. Clawing object.                                                                                                                                                                                                                                                      | Low                    | Low                   |
| **Interest**    | Dilated/round pupils. Gaze to right. Observing an individual or object.                                                                                  | Upright and directed forward towards stimulus. Ear flick.  | Depends on context. Horizontal. Tail up/vertical in friendly greeting.                                                                     | Standing on hindlimbs. Resting forepaws against object. Stretching head out forward. Head turn to right.         | Exploring the area or objects. Sniffing. Licking. Pawing. Friendly greeting (touching noses with another cat or rubbing face & body against object/individual-allorubbing). Hunting (stalking, chasing, pouncing, grabbing, biting).                                                                                                                                                                                                                                                                                                      | Moderate               | Moderate              |

> **Note:** Pupil size and shape may also be influenced by arousal and ambient light levels.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7701e8af-2bf2-4b75-b0e3-299ff16c839e" width="400" height=800 title="feline-emotions-images">
</p>

### From the Ethogram to Computer Vision

The ethogram gave me something much more useful than simply a list of emotions: it gave me **observable characteristics that could potentially be translated into measurable features**.

For example:

* **Eyes:** eye shape and degree of constriction
* **Ears:** ear orientation
* **Mouth:** mouth shape and opening
* **Body and tail:** additional behavioral cues that can provide context

However, not all of these characteristics can be reliably extracted from a single facial photograph. My computer vision pipeline therefore focuses primarily on the facial features that can be detected and measured from landmarks.

This led to an important design decision: rather than treating one facial characteristic as a direct indicator of an emotion, I would combine multiple measurements to infer the cat's emotional state.

For example, an eye characteristic alone does not necessarily mean that a cat is afraid or content. Pupil size can be affected by lighting and arousal, while ear position and mouth shape can provide additional context.

The goal was therefore to turn the qualitative observations from the ethogram into **quantitative, landmark-based features**.

### From Research to the MoodMeow Labels

The five emotional states from the ethogram became the foundation for the five categories used by MoodMeow:

| Research framework | MoodMeow    |
| ------------------ | ----------- |
| Fear               | **Afraid**  |
| Anger/Rage         | **Angry**   |
| Joy/Play           | **Playful** |
| Contentment        | **Happy**   |
| Interest           | **Curious** |

These names are intentionally simplified for the app, but the underlying categories are based on the emotional states described in the research.

Importantly, MoodMeow is not intended to definitively measure a cat's internal emotional state. Instead, it uses observable facial characteristics as signals from which an emotional state is **inferred**.

With this research foundation in place, the next challenge was to determine how to detect and measure these facial characteristics automatically.

That brought me to the next stage of the project: **finding a suitable dataset and developing a facial landmark detection pipeline.**

---
#### Resources
- Nicholson, S.L., O’Carroll, R.Á. Development of an ethogram/guide for identifying feline emotions: a new approach to feline interactions and welfare assessment in practice. Ir Vet J 74, 8 (2021). https://doi.org/10.1186/s13620-021-00189-z
