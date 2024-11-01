---
title: Select and Training ML model
parent: MLOps
nav_order: 2
layout: default
---

A machine learning model development process consists of three main components: the algorithm (or model), hyperparameters (parameters that control the learning process), and data. As discussed in the post on [ML deployment](https://seyong2.github.io/portfolio/docs/data-science/mlops/deployment.html) deploying an ML model is iterative, requiring constant monitoring to ensure smooth performance. Similarly, the process of developing a machine learning algorithm is also highly iterative. 

We start with an initial model, hyperparameters, and data, followed by training and error analysis to assess performance. Based on this analysis, we refine the model, hyperparameters, or data. We repeat this cycle until the model's performance is satisfactory. Before production deployment, a final audit ensures that the model is robust and ready.

<p align="center">
  <img src=https://github.com/user-attachments/assets/57e70a55-2c6d-45a9-adc7-65646f1dc6d4>
</p>

## Low Test Error is NOT Enough

When testing an ML model, three essential criteria must be met. First, the model should perform well on the training dataset, typically measured by average training error. This step may seem straightforward, as the learning process aims to minimize the gap between ground truth and model predictions. However, achieving good performance on training data alone is insufficient; the model must also generalize well to validation and test datasets, showing robust performance on data it hasn't encountered before.

Over recent decades, ML development has often focused on optimizing hold-out dataset performance. However, in many cases, models must also align with business metrics and project goals to be considered successful.

Some situations illustrate how a model with low average test error might still underperform in practical applications.

### Example 1: Web search

Consider web search queries, which can be either informational (seeking answers or specific information) or navigational (seeking a specific site or page). For navigational queries, users expect exact results, as they’re trying to locate a particular destination. If a search engine fails on even a small number of these critical queries, it could be unacceptable for deployment, even if the model achieves low test error overall. This is because navigational queries represent a disproportionately important subset where precision is crucial.

### Example 2: Loan approval

Imagine an ML algorithm designed to assist with loan approvals. Here, it’s essential that the model doesn’t unfairly discriminate based on protected attributes like gender, ethnicity, profession, or address. A model with high accuracy on the test set would still be unsuitable if it exhibits bias, as this could lead to ethical and legal concerns.

### Example 3: Medical diagnosis

In medical diagnosis, it’s common to work with datasets that are highly imbalanced (e.g., 99% negative cases for a particular disease). If a model simply predicts “negative” for all cases, it could achieve 99% accuracy, but it would fail critically when diagnosing a patient who actually has the disease. Such a system could be unreliable and dangerous, emphasizing the need for metrics beyond average test error.

Thus, while good performance on a test dataset is important, it’s crucial to ensure that the model meets the specific needs and goals of its intended application. 

## Establishing Baseline

How can we tell if an error rate is low or high? We start by establishing a baseline, providing a reference level to improve upon. The approach for setting a baseline varies depending on whether the data is unstructured or structured.

Unstructured data (e.g., text and multimedia) lacks a predefined format or schema, making human-level performance (HLP) a reasonable baseline, as humans are typically proficient at interpreting this type of data.

For structured data (e.g., tables with clearly defined attributes), which computers process efficiently, we might look at prior research for comparison. Another option is to quickly implement a simple model without extensive optimization as a starting point. Finally, when replacing an older system, comparing the new model’s performance to the old one is valuable.

A baseline indicates what’s feasible and can reveal irreducible error (Bayes error), helping us prioritize improvements more effectively.

## Additional Tips

- **Literature Review**: Explore what’s achievable (e.g., through courses, blogs, or open-source projects). Avoid chasing the latest algorithms; often, a reasonable algorithm with good data outperforms a state-of-the-art algorithm with inadequate data.

- **Deployment Constraints**: When selecting a model, consider deployment requirements, such as computing resources, if the goal is implementation. However, if the aim is to establish a baseline and explore possibilities, performance constraints are less critical.

- **Sanity Check for Code and Algorithm**: Test the model’s ability to overfit on a small dataset before training on larger sets. This strategy can reveal issues quickly, often within minutes or seconds, aiding in faster debugging.

#### Resources 
- [Machine Learning in Production by DeepLearning AI](https://www.coursera.org/learn/introduction-to-machine-learning-in-production/home/week/1)
