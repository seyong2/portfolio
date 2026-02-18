---
title: Why Extreme? - Behind XGBoost
parent: Computer Vision
nav_order: 8
layout: default
---

If we look at the notebooks of the top rankers in many machine learning competitions, it is easy to notice that they often use XGBoost to achieve high scores. So what makes XGBoost such a popular choice among competition participants? In this post, we will explore what sets XGBoost apart from other algorithms.

**XGBoost** (eXtreme Gradient Boosting) is an advanced implementation of the traditional gradient boosting algorithm, which builds a sequential ensemble of weak learners. These weak learners are typically decision trees, and each new tree attempts to correct the mistakes made by the previous ones. 

As its name suggests, XGBoost's strength comes from leveraging second-order gradient information, advanced regularization techniques, and highly efficient computational implementations. These features make it especially attrative for problems that require both high predictive accuracy and computational efficiency. 

<figure>
  <p align="center">
  <img width="850" height="339" alt="image" src="https://github.com/user-attachments/assets/9407d188-986d-4427-ae6a-18092bdd7747" />
  <figcaption><p align="center">XGBoost in visualization</p></figcaption>
  </p>
</figure>

# Mathematical Framework

At each iteration $$t$$, XGBoost aims to achieve two objectives: minimize prediction error and prevent overfitting. Accordingly, the general form of the objective function at the $$t$$-th iteration can be defined as:

$$ L^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) $$

where 
- $$n$$ is the number of observations in the training data,
- $$l(\cdot,\cdot)$$ is a differentiable loss function,
- $$y_i$$ is the true label for instance $$i$$,
- $$\hat{y}_i^{(t-1)}$$ is the prediction label for instance $$i$$ after $$t-1$$ trees,
- $$f_t(x_i)$$ is the output produced by a new tree at iteration $$t$$, and
- $$\Omega(f_t) is a regularization term that penalizes the complexity of the new tree.

## Second-Order Taylor Expansion

Unlike traditional gradient boosting, XGBoost approximates the loss function using a second-order Taylor expansion. By incorporating both the gradient (first derivative) and the curvature (second derivative) of the loss, XGBoost achieves a more accurate approximation of the obejctive function. This, in turn, leads to more precise optimization steps and faster convergence.

To refresh our memory, the second-order Taylor expansion (or quadratic approximation) of a function $$f(x+\Delta x)$$ approximates the function using its first two derivatives:

$$ f(x+\Delta x) \approx f(x) + \frac{df(x)}{dx}\Delta x + \frac{d^2f(x)}{dx^2}\frac{(\Delta x)^2}{2!} $$

Keeping this in mind, the loss function around the current prediction $$\hat{y}_i^{(t-1)}$$ for instance $$i$$ can be approximated as follows:

$$ l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) $$

where

- $$g_i$$ is the first-order gradient, defined as $$ g_i = \frac{\delta l(y_i, \hat{y}_i^{(t-1)})}{\delta \hat{y}_i^{(t-1)} }$$ . It measures how much the loss changes with a small change in the prediction.
- $$h_i$$ is the second-order gradient (Hessian), defined as $$ h_i = \frac{\delta^2 l(y_i, \hat{y}_i^{(t-1)})}{\delta (\hat{y}_i^{(t-1)})^2 }$$ . It measures how the gradient itself changes, i.e., the curvature of the loss function.

By using both first- and second-order information- not just the gradient- we can make more precise and stable updates.

Plugging this approximation into the original objective function, we obtain:

$$ L^(t) \approx \sum_i^n \[l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \] + \Omega(f_t) $$

Since $$ l(y_i, \hat{y}_i^{(t-1)}) $$ does not depend on $$f_t$$, it can be treated as a constant during optimization and therefore omitted. The objective simplifies to: 

$$ L^(t) \approx \sum_i^n \[g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \] + \Omega(f_t) $$

https://mbrenndoerfer.com/writing/xgboost-extreme-gradient-boosting-complete-guide-mathematical-foundations-python-implementation

#### Resources
- [Identifying the Signature of Suicidality : A Machine Learning Approach](https://www.sciencedirect.com/science/article/abs/pii/S0165032715310922?via%3Dihub)
