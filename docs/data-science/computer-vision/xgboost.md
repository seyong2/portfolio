---
title: Why Extreme? - Behind XGBoost
parent: Computer Vision
nav_order: 8
layout: default
---

If we look at the notebooks of the top rankers in many machine learning competitions, it is not hard to find that they used XGBoost algorithm to achieve those high scores. Then, what is behind XGBoost so that it has become a go-to choice for many competition participants? In this post, we're going to explore what makes XGBoost separate from other algorithms.

XGBoost (eXtreme Gradient Boosting) is a variant of traditional gradient boosting algorithm, which is built on a sequential ensemble of weak learners. These weak learners typically are decision trees and each new tree corrects mistakes made by the previous ones. However, as its name suggests, its power comes from the fact that it leverages on second-order gradient information, advanced regularization techniques, and highly efficient computational implementations. Then, these factors make XGBoost attrative for problems that require both accuracy and computational efficiency. 

- add a graph of XGBoost

# Mathematical Framework

There are two obectives XGBoost tries to achieve at each iteration $$t$$: minimize prediction error and avoid overfitting. Then, the general form of the objective function at $$t$$-th iteration can be defined as:

$$ L^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) $$

where 
- $$n$$ is the number of observations in the training data,
- $$l(\cdot,\cdot)$$ is a differentiable loss function,
- $$y_i$$ is the true label for instance $$i$$,
- $$\hat{y}_i^{(t-1)}$$ is the prediction label for instance $$i$$ after $$t-1$$ trees,
- $$f_t(x_i)$$ is the output of a new tree being added at iteration $$t$$, and
- $$\Omega(f_t) is a regularization term that penalizes the complexity of the new tree.

## Second-Order Taylor Expansion

Unlike the traditional gradient boosting, XGBoost approximates the loss function using the second-order Taylor expansion. By using both the gradient (first derivative) and the curvature (second derivative) of the loss, XGBoost can achieve more accurate approximation of the obejctive function, which can be translated into better optimization steps and faster convergence.

### Write another formula
To refresh our memories...
The second-order Taylor expansion (or quadratic approximation) of a function $$f(x)$$ around a point $$a$$ approximates the function using its first two derivatives:

$$ f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2}(x-a)^2 $$

Keeping this in mind, the loss function around the current prediction $$\hat{y}_i^{(t-1)}$$ for instance $$i$$ can be approximated as follows:

$$ l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx l(y_i, \hat{y}_i^{(t-1)}) + 

https://mbrenndoerfer.com/writing/xgboost-extreme-gradient-boosting-complete-guide-mathematical-foundations-python-implementation
