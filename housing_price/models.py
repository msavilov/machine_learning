from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

import preprocessing


def model(data, labels):
    forest_reg = make_pipeline(preprocessing,
                               RandomForestRegressor(random_state=42))
    forest_rmses = -cross_val_score(forest_reg, data, labels,
                                    scoring="neg_root_mean_squared_error", cv=10)
