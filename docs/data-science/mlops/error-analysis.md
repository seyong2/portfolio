---
title: Error Analysis and Performance Auditing
parent: MLOps
nav_order: 3
layout: default
---

### What is Error Analysis?

Machine learning models rarely work perfectly on their first iteration. Error analysis is a key process for systematically identifying where a model struggles and finding the most efficient ways to improve its performance. Since training and deploying machine learning models are iterative processes, error analysis should be iterative as well. Let’s go through what error analysis entails using an example from a speech recognition system.

Suppose we tested our speech recognition model on a validation dataset and found 100 mislabeled audio examples. To understand these errors better, we can categorize them by potential error sources or "tags," such as "car noise" or "people noise." By listening to each mislabeled example, we can assign one or more tags to capture the types of noise that contributed to the errors. For example, we might discover that "low bandwidth" also affects transcription accuracy, and we can retroactively tag examples that fit this new category. This iterative tagging process helps us systematically understand the root causes of errors, as shown in the figure below. By the end, we’ll have a structured analysis that reveals which error sources are most common, guiding us to focus improvement efforts where they’re likely to make the biggest impact.

<p align="center">
  <img src="https://github.com/user-attachments/assets/188f1eb2-109c-4cab-8388-4f5234653d57">
</p>

Until now, manual error analysis has often been done using spreadsheets or Jupyter notebooks, but new MLOps tools are emerging to streamline this process for developers.

### Useful Metrics for Error Analysis

Error analysis metrics help us assess which issues to prioritize. Here are some useful metrics when analyzing model errors:

- **What percentage of errors have a specific tag?**
Example: In our speech recognition example, if 12% of mislabeled clips have "car noise" and 5% have "people noise," we might prioritize addressing car noise because it’s a larger source of error.

- **What percentage of data with a specific tag is mislabeled?**
Example: If 18% of all audio clips containing car noise are mislabeled, we know that car noise presents a particularly challenging transcription problem.

- **What fraction of the entire dataset contains a specific tag?**
This metric helps us understand the prevalence of each category relative to the whole dataset, helping prioritize issues that affect a larger portion of data.

- **How much room for improvement exists for each tag?**
By comparing human performance to model performance on specific tags (like car noise), we can estimate the maximum improvement possible for the model.

- **How feasible is it to improve performance for each tag?**
If there’s an accessible technique, such as an external system to isolate car noise, we might prioritize addressing that error type, assuming it has significant impact.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9640f1bc-73cd-4393-adad-9e55892a6757">
</p>

After identifying which tags to address, you may consider adding more data, augmenting existing data, or enhancing label accuracy to improve model performance on those specific tags.

### How to Handle Skewed Datasets

For highly imbalanced datasets, traditional metrics like raw accuracy can be misleading. Imagine a medical diagnosis dataset where only 1% of patients have a disease. An algorithm that always predicts "no disease" would achieve 99% accuracy, which isn’t useful in this context. For such skewed data, a confusion matrix is more informative.

<p align="center">
  <img src="https://github.com/user-attachments/assets/af065c5d-9a7d-411a-bcac-e1feac617dd7">
</p>

In a confusion matrix, we track:

- **True Positives (TP)**: Correct positive predictions.
- **True Negatives (TN)**: Correct negative predictions.
- **False Positives (FP)**: Incorrectly predicted positives.
- **False Negatives (FN)**: Incorrectly predicted negatives.

From these values, we can derive two key metrics:

- **Precision ($$\frac{TP}{(TP + FP)}$$)**: The fraction of correctly predicted positives among all positive predictions.
- **Recall ($$\frac{TP}{(TP + FN)}$$)**: The fraction of actual positives correctly identified by the model.

For example, if an algorithm predicts only "negative" results, its recall will be 0, indicating it misses all positive examples, regardless of precision. Ideally, both precision and recall should be high. To compare models that trade off between these metrics, we use the $$F_1$$ score, a metric that combines precision and recall:

$$F_1=\frac{2}{\frac{1}{Precision}+\frac{1}{Recall}}$$

The $$F_1$$ score penalizes models with imbalanced precision and recall. As shown in the example below, low precision or recall will result in a low F1 score, highlighting the importance of balancing these metrics.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4a173ef1-ed6b-47d7-9233-e841908012b9">
</p>

Precision and recall are also helpful in multi-class settings. Suppose you’re classifying rare manufacturing defects in smartphones, such as scratches, dense marks, or screen discolorations. Here, it’s often critical to achieve high recall so that defective units don’t reach customers. Slightly lower precision may be acceptable if flagged items can be reviewed manually.

In these cases, the F1 score provides a single metric to evaluate model performance across rare classes. Rather than focusing on overall accuracy, which could be high even with missed defects, the $$F_1$$ score helps prioritize model improvements that impact rare but significant issues.

### Performance Auditing

Even when a model achieves a high accuracy or $$F_1$$ score, it’s important to conduct a performance audit before deployment. This step can prevent issues post-deployment by catching overlooked errors, fairness concerns, or bias in the model’s predictions.

1. **Identify Potential Problems**: Consider the ways in which your model might fail or exhibit bias, particularly on data subsets like gender or ethnicity. It’s useful to track the occurrence of specific errors like FP or FN rates on critical subsets.

2. **Define Auditing Metrics**: Establish metrics to measure model performance on these targeted issues. MLOps tools can help automate this evaluation for each model, making it easier to track performance on relevant data slices.

3. **Gain Stakeholder Buy-in**: Collaborate with business and product teams to agree on the problems and metrics that are critical to address before deployment. This ensures alignment on the criteria for a successful model launch.

Tools like TensorFlow Model Analysis (TFMA) offer options for evaluating model metrics across different slices of data, helping you spot potential issues before they escalate post-deployment. By identifying and addressing potential problems now, you can improve model reliability and reduce the chance of adverse outcomes later.

In summary, error analysis is a structured approach to diagnosing and improving machine learning models. By using targeted metrics, handling skewed data with precision and recall, and performing thorough audits, you can enhance model accuracy and resilience, ensuring a smoother deployment and operation in real-world settings.

#### Resources
- [Machine Learning in Production by DeepLearning AI](https://www.coursera.org/learn/introduction-to-machine-learning-in-production/home/week/2)
