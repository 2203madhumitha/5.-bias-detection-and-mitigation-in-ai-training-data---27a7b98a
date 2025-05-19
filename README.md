# Bias Detection and Mitigation in AI Training Data

A web-based tool to detect and mitigate bias in machine learning datasets using Fairlearn and Scikit-learn. The app allows users to upload a CSV file, choose sensitive attributes and target labels, and analyze fairness metrics before and after bias mitigation.

**Live Demo**:  
[https://five-bias-detection-and-mitigation-in-ai.onrender.com](https://five-bias-detection-and-mitigation-in-ai.onrender.com)

---

## Features

- Upload any CSV dataset for fairness analysis
- Select **binary sensitive attributes** (e.g., gender,disability) and **binary target labels** (e.g., hired/not hired)
- View bias metrics:
  - Demographic Parity Difference
  - Equalized Odds Difference
- Automatically apply in-processing bias mitigation using **Fairlearn**
- Visual comparison of prediction rates before and after mitigation
- Model accuracy comparison (original vs. debiased)

---

## What This App Does

Bias in machine learning models can cause unfair decisions based on race, gender, or other protected attributes. This app helps:

- Detect bias in your AI training dataset
- Reduce it using fairness algorithms
- Visualize disparities between groups

