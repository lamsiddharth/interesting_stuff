import numpy as np

# Feature data: each row is one observation and each column is one feature.
X = np.array([
    [20, 100],
    [30, 115],
    [40, 120],
    [60, 145],
    [70, 150],
    [80, 165]
])

# Class label for each observation in X.
y = np.array([0, 0, 0, 1, 1, 1])

# Separate the observations into their respective classes.
X0 = X[y == 0]
X1 = X[y == 1]

print("X0:")
print(X0)
print("X1:")
print(X1)

# Calculate the prior probability of each class:
# number of observations in the class / total number of observations.
prior_0 = X0.shape[0] / X.shape[0]
prior_1 = X1.shape[0] / X.shape[0]

print("Prior probability of class 0:", prior_0)
print("Prior probability of class 1:", prior_1)

# Calculate the mean of each feature for each class.
# axis=0 calculates one mean for each column.
mu0 = np.mean(X0, axis=0)
mu1 = np.mean(X1, axis=0)
print("Mean of class 0:", mu0)
print("Mean of class 1:", mu1)

# Calculate the covariance matrix for each class.
# rowvar=False means columns are variables/features and rows are observations.
# bias=True uses the population covariance formula, dividing by N.
sigma0 = np.cov(X0, rowvar=False, bias=True)
sigma1 = np.cov(X1, rowvar=False, bias=True)
print("Covariance matrix of class 0:")
print(sigma0)
print("Covariance matrix of class 1:")
print(sigma1)


def gaussian_density(x, mu, sigma):
    """Calculate the multivariate Gaussian density of x."""
    # Number of features in the observation.
    size = len(x)

    # Determinant of the covariance matrix, used in the normalizing constant.
    det_sigma = np.linalg.det(sigma)

    # Normalizing constant from the multivariate Gaussian formula.
    norm_const = 1.0 / (np.power((2 * np.pi), float(size) / 2) * np.sqrt(det_sigma))

    # Difference between the observation and the class mean.
    x_mu = x - mu

    # Inverse covariance matrix, used to calculate the Mahalanobis distance.
    inv_sigma = np.linalg.inv(sigma)

    # Exponential part of the multivariate Gaussian formula.
    result = np.exp(-0.5 * (x_mu @ inv_sigma @ x_mu.T))

    # Return the probability density for x under this class distribution.
    return norm_const * result

# First new observation to classify.
x_new = np.array([50, 130])

# Calculate its density under both class distributions.
gda0 = gaussian_density(x_new, mu0, sigma0)
gda1 = gaussian_density(x_new, mu1, sigma1)

print("Gaussian density for class 0:", gda0)
print("Gaussian density for class 1:", gda1)

# Predict the class with the greater Gaussian density.
prediction = 0 if gda0 > gda1 else 1

print("Predicted class:", prediction)

# Second new observation to classify.
x_new_1 = np.array([75, 155])

# Calculate its density under both class distributions.
gda0 = gaussian_density(x_new_1, mu0, sigma0)
gda1 = gaussian_density(x_new_1, mu1, sigma1)

print("Gaussian density for class 0:", gda0)
print("Gaussian density for class 1:", gda1)

# Predict the class with the greater Gaussian density.
prediction = 0 if gda0 > gda1 else 1
print("Predicted class:", prediction)
