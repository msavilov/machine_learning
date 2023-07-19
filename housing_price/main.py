from load_data import load_housing_data
from sklearn.model_selection import train_test_split


def main():
    data = load_housing_data()
    train_set, test_set = train_test_split(data, 0.2)


if __name__ == '__main__':
    main()
