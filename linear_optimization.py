import numpy as np

class MathOptimizer:
    """
    A mathematical approach to Linear Regression using 
    Ordinary Least Squares (OLS).
    """
    def _init_(self):
        self.slope = None
        self.intercept = None

    def fit(self, x, y):
        # Calculating the mean of X and Y
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Mathematical formula for Slope (beta1): 
        # sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean)**2)
        
        self.slope = numerator / denominator
        self.intercept = y_mean - (self.slope * x_mean)
        
        print(f"Model Optimized. Slope: {self.slope:.4f}, Intercept: {self.intercept:.4f}")

# Example Dataset (Input features and target values)
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

model = MathOptimizer()
model.fit(X, Y)
