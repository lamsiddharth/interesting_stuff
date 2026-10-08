import numpy as np


# -----------------------------
# 1. Sigmoid
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -----------------------------
# 2. Logistic Regression
# -----------------------------
class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        self.loss_history = []

    # -------------------------
    # 3. Train the model
    # -------------------------
    def fit(self, X, y):

        n, d = X.shape

        # Initialize parameters
        self.w = np.zeros(d)
        self.b = 0.0

        for epoch in range(self.epochs):

            # ---------------------
            # Forward pass
            # ---------------------

            # z = Xw + b
            z = X @ self.w + self.b

            # p = sigmoid(z)
            p = sigmoid(z)

            # ---------------------
            # Calculate loss
            # ---------------------

            loss = -np.mean(
                y * np.log(p) +
                (1 - y) * np.log(1 - p)
            )

            self.loss_history.append(loss)

            # ---------------------
            # Backward pass
            # ---------------------

            # dw = (1/n) X^T(p-y)
            dw = (X.T @ (p - y)) / n

            # db = mean(p-y)
            db = np.mean(p - y)

            # ---------------------
            # Gradient descent
            # ---------------------

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # Print progress
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}, Loss: {loss:.4f}"
                )

    # -------------------------
    # 4. Predict probabilities
    # -------------------------
    def predict_proba(self, X):

        z = X @ self.w + self.b

        return sigmoid(z)

    # -------------------------
    # 5. Predict classes
    # -------------------------
    def predict(self, X):

        p = self.predict_proba(X)

        return (p >= 0.5).astype(int)


# =================================
# Example dataset
# =================================

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9]
], dtype=float)

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])


# =================================
# Train
# =================================

model = LogisticRegression(
    learning_rate=0.1,
    epochs=1000
)

model.fit(X, y)


# =================================
# Predictions
# =================================

probabilities = model.predict_proba(X)
predictions = model.predict(X)

print("\nProbabilities:")
print(probabilities)

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y)

print("\nLearned weights:")
print(model.w)

print("\nLearned bias:")
print(model.b)