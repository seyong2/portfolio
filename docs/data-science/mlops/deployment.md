---
title: ML Deployment
parent: MLOps
nav_order: 1
layout: default
---

Deploying machine learning (ML) models involves unique challenges that fall into three main areas: **model and statistical issues, deployment cases, and monitoring**. These challenges go beyond just building a working model—they require robust planning for adaptability, reliability, and long-term effectiveness in production.

### 1. Issues: Statistical and Software Challenges

A major ML challenge during deployment is managing **data drift** and **concept drift**. Data drift refers to changes in the input data distribution, which can degrade model performance if not addressed. For example, a model predicting stock prices may underperform if new economic conditions cause stock patterns to differ from the training data. Concept drift, on the other hand, occurs when the actual relationship between input and output shifts, making previous patterns obsolete. For instance, if an online retail model used to predict customer purchases sees that preferences shift toward sustainable products, the original prediction logic may no longer apply, necessitating a model update.

On the software side, choices around real-time vs. batch processing, compute resources (e.g., CPU vs. GPU), and deployment location (cloud, edge, or browser-based) all impact the final design. Real-time models require low latency, while batch processing can handle larger volumes without demanding immediate responses. Other considerations include logging, security, and privacy to ensure model transparency and data safety, especially for sensitive or regulated applications.

### 2. Deployment Cases: Strategies for Implementation

There are three main scenarios for deploying an ML model: **introducing a new capability, assisting with or automating manual tasks, and replacing an existing ML system**. 

In the first case, a common approach is to direct a small amount of traffic to the model initially and gradually increase it as performance is validated. This strategy ensures that any issues are caught early with minimal disruption. For assisting with or automating tasks, a useful approach is **shadow mode**, where the ML model operates alongside a human without impacting decisions initially. This setup allows for data collection and performance monitoring without affecting outcomes, so the model’s reliability can be assessed. Finally, when replacing an existing ML model, **canary deployments** or **blue-green deployments** are common patterns. A canary deployment directs a fraction of traffic to the new model, allowing for gradual scaling and easy rollback if issues arise. Blue-green deployments involve keeping the old model live while switching traffic to the new one, offering a straightforward rollback option if needed.

### 3. Monitoring: Metrics and Iteration
Monitoring deployed models is essential to maintain quality over time, and dashboards are often used to track system performance. The first step is to consider what could go wrong with the model or data, and then choose key metrics to detect such issues. Important metrics include software-related measures (e.g., memory usage, latency), input data metrics (e.g., average input length or missing values), and output metrics (e.g., error rates or the number of retries). Monitoring for drift is also key: data drift and concept drift can be detected by comparing input data distributions or tracking shifts in model accuracy.

Once metrics are in place, setting thresholds for alerts helps catch any concerns early. This monitoring process is iterative, evolving as the model gathers data in production. When maintenance is needed, many organizations rely on **manual retraining**, where an engineer evaluates the model’s performance on new data. Although automated retraining systems exist, they are less common, as many prefer human oversight before making significant changes.

In summary, deploying an ML model effectively involves addressing data and statistical issues, choosing appropriate deployment patterns, and establishing a robust monitoring system. By iterating and refining these strategies, models remain accurate, reliable, and aligned with real-world changes.

#### Resources
- [Machine Learning in Production by DeepLearning AI](https://www.coursera.org/learn/introduction-to-machine-learning-in-production/home/week/1)
