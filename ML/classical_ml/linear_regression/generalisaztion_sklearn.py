from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

X, y = make_moons(
    n_samples=300,
    noise=0.25,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

degrees = [1, 2, 5, 15, 20]

for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree),
        LogisticRegression(max_iter=1000000)
    )

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)

    print(
        degree,
        train_acc,
        val_acc
    )