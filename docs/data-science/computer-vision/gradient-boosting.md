---
title: Why Extreme? - Behind XGBoost
parent: Computer Vision
nav_order: 8
layout: default
---

If we look at the notebooks of the top rankers in many machine learning competitions, it is not hard to find that they used XGBoost algorithm to achieve those high scores. Then, what is behind XGBoost so that it has become a go-to choice for many competition participants? In this post, we're going to explore what makes XGBoost separate from other algorithms.

XGBoost (eXtreme Gradient Boosting) is a variant of traditional gradient boosting algorithm, which is built on a sequential ensemble of weak learners. These weak learners typically are decision trees and each new tree corrects mistakes made by the previous ones. However, as its name suggests, its power comes from the fact that it leverages on second-order gradient information, advanced regularization techniques, and highly efficient computational implementations. Then, these factors make XGBoost attrative for problems that require both accuracy and computational efficiency. 

- add a graph of XGBoost


https://mbrenndoerfer.com/writing/xgboost-extreme-gradient-boosting-complete-guide-mathematical-foundations-python-implementation
