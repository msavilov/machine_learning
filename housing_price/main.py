from load_data import load_housing_data
from imputer import imputer_func
from sklearn.model_selection import train_test_split


def main():
    data = load_housing_data()
    data = imputer_func(imputer_func, 'median')
    train_set, test_set = train_test_split(data, 0.2)


if __name__ == '__main__':
    main()
