from sklearn.impute import SimpleImputer


def imputer_func():
    imputer = SimpleImputer(strategy='median')
    return imputer
