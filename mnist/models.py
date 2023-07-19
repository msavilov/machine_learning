from sklearn.linear_model import SGDClassifier


def classified(X_train, y_train):
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)
    return sgd_clf
