---
title: Linear Interpolation in Time Series
parent: Time Series
nav_order: 1
layout: default
---

There are many situation when working with time series data where we encounter missing values between known data points. Interporation is a technique commonly used in time series analysis to estimate these missing data by leveraging existing data points. This allows us to create a continuous and complete representation of the time series.

There are many advanced interpolation techniques available, but in this post I want to focus on **linear interpolation**, which is one of the most commonly used methods for handling missing values in time series data. 

Linear interpolation is technique that assumes the relationship between data points is linear in nature. It estimates unknown values by drawing a straight line between two known points and using that line to approximate the missing values.

To calculate an unknown value $$y$$ at a specific time $$x$$ between two known points $$(x_1, y_1)$$ and $$(x_2, y_2)$$, the linear interpolation formula is: 

$$ y = y_1 + \frac{y_2-y_1}{x_2-x_1} (x-x_1)$$

This formula calculates a weighted estimate based on the proximity of the target time $$x$$ to the known timestamps $$x_1$$ and $$x_2$$.

Let's look at a simple example using an oil price dataset where some dates have missing oil prices.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f5dfc077-dcf1-4148-9748-7692b0261be6">
</p>

In Python, we can apply the linear interploation method discussed above using the 
`interpolate(method='linear')` function.

```python
df_oil['dcoilwtico_interpolated'] = df_oil['dcoilwtico'].interpolate(method='linear')
```

If we plot the interpolated oil prices (in red), we can observe that the gaps are no longer present.

<p align="center">
  <img src="https://github.com/user-attachments/assets/af21d0c5-b7dd-4c87-861f-0ac6c8d21e6a">
</p>

---
