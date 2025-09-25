# Linear Regression Project

## Overview
This project demonstrates **Simple and Multiple Linear Regression** using Python.  
The goal is to predict **house prices** based on input features like median income, house age, etc.  
It includes **data preprocessing, model training, evaluation, and visualization**.

---

## Dataset
- **California Housing Dataset** from `sklearn.datasets`.  
- Features include:
  - `MedInc` – Median income in the block
  - `HouseAge` – Median house age
  - `AveRooms` – Average rooms per household
  - `AveOccup` – Average occupants per household
  - `Latitude`, `Longitude`, etc.  
- Target:
  - `Price` – Median house price

---

## Tools & Libraries
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Project Steps

1. **Import Libraries**
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
   from sklearn.datasets import fetch_california_housing
