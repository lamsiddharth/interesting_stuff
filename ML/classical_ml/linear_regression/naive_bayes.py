#naive bayes implementation
import numpy as np

X = np.array([
    [1, 1], #these are counts of words in emails, for example "urgent" and "money"
    [1, 0],
    [0, 1],
    [0, 0],
    [1, 0],
    [0, 0],
])

y = np.array([
    "spam",
    "spam",
    "spam",
    "not_spam",
    "not_spam",
    "not_spam",
])

class CategoricalNBFFromScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.classes_ = None
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_, class_counts = np.unique(
            y, return_counts=True
        )

        n_samples, n_features = X.shape

        for cls, count in zip(self.classes_, class_counts):
            # Calculate prior probability for each class
            self.class_log_prior_[cls] = np.log(count / n_samples)

        for cls in self.classes_:
            # Select samples belonging to the current class
            X_cls = X[y == cls]
            n_cls = len(X_cls)

            for j in range(n_features):
                for values in [0, 1]:  # Assuming binary features
                    count = np.sum(X_cls[:, j] == values)
                    # Apply Laplace smoothing
                    smoothed_count = count + self.alpha
                    smoothed_total = n_cls + 2 * self.alpha  # 2 for binary features
                    self.feature_log_prob_[(cls, j, values)] = np.log(
                        smoothed_count / smoothed_total
                    )
        return self

    def predict_log_scores(self, X):
        x = np.asarray(X)
        n_samples, n_features = x.shape

        scores = np.zeros((n_samples, len(self.classes_)))

        for i in range(n_samples):
            for j, cls in enumerate(self.classes_):
                log_score = self.class_log_prior_[cls]
                for k in range(n_features):
                    feature_value = x[i, k]
                    log_score += self.feature_log_prob_.get(
                        (cls, k, feature_value), np.log(self.alpha / (self.alpha * 2))
                    )
                scores[i, j] = log_score
        return scores

    def predict(self, X):
        scores = self.predict_log_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]
    
model = CategoricalNBFFromScratch(alpha=1.0)
model.fit(X, y)

X_test = np.array([
    [1, 1],
    [0, 1],
    [1, 0],
    [0, 0],
])

print("Classes:", model.classes_)
print("Log-scores:\n", model.predict_log_scores(X_test))
print("Predictions:", model.predict(X_test))

