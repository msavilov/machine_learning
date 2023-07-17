from load_data import load_housing_data
from shuffle_and_split_data import shuffle_and_split


def main():
    housing = load_housing_data()
    train_set, test_set = shuffle_and_split(housing, 0.2)


if __name__ == '__main__':
    main()
