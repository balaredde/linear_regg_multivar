# linear_regg_multivar

# 📊 Linear Regression - Understanding the Best Fit Line  

## 📌 Introduction  
Linear Regression is a fundamental machine learning algorithm used to model the relationship between a dependent variable (\( y \)) and one or more independent variables (\( x \)). The objective is to find the best-fit line that minimizes the error between predicted and actual values.

The equation of the line is:  
\[
y = mx + c
\]  
where:  
- \( y \) = predicted output (dependent variable)  
- \( x \) = input feature (independent variable)  
- \( m \) = slope of the line  
- \( c \) = y-intercept  

---

## 🔢 How the Best-Fit Line is Taken  

### 1️⃣ **Initializing a Random Line**  
We start by assuming a random line with initial values of \( m \) and \( c \), which are not yet optimized.

### 2️⃣ **Measuring the Error (Cost Function - MSE)**  
To determine how well the line fits, we calculate the **Mean Squared Error (MSE):**  
\[
MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
\]  
where:  
- \( y_i \) = actual value  
- \( \hat{y}_i \) = predicted value  
- \( n \) = number of data points  

### 3️⃣ **Optimizing the Line (Finding Best \( m \) and \( c \))**  
We adjust \( m \) and \( c \) to minimize the error using:  

#### ✅ **Least Squares Method (Direct Calculation)**
\[
m = \frac{n \sum (x_i y_i) - \sum x_i \sum y_i}{n \sum x_i^2 - (\sum x_i)^2}
\]
\[
c = \frac{\sum y_i - m \sum x_i}{n}
\]

#### ✅ **Gradient Descent (Iterative Approach)**
\[
m = m - \alpha \frac{\partial MSE}{\partial m}
\]
\[
c = c - \alpha \frac{\partial MSE}{\partial c}
\]

where \( \alpha \) is the learning rate.

---

## 🚀 **Python Code for Linear Regression**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data
X = np.array([1, 2, 3, 4]).reshape(-1, 1)  # Independent variable (2D)
y = np.array([2, 3, 5, 7])  # Dependent variable (1D)

# Creating the model
model = LinearRegression()
model.fit(X, y)  # Train the model (Find best-fit line)

# Predict values
predicted_y = model.predict(X)

# Plot the data
plt.scatter(X, y, color='red', label="Actual Data")
plt.plot(X, predicted_y, color='blue', label="Best-Fit Line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression - Best Fit Line")
plt.show()
