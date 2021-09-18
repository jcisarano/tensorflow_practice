# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

DATA_SERVER_ROOT: str = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
LOCAL_SAVE_PATH: str = os.path.join("datasets", "housing")
REMOTE_DATA_URL: str = DATA_SERVER_ROOT + "datasets/housing/housing.tgz"
LOCAL_ZIP_FILENAME: str = "housing.tgz"
LOCAL_CSV_FILENAME: str = "housing.csv"


def fetch_remote_data(remote_data_url=REMOTE_DATA_URL, local_save_path=LOCAL_SAVE_PATH,
                      local_filename=LOCAL_ZIP_FILENAME):
    os.makedirs(local_save_path, exist_ok=True)
    full_local_path = os.path.join(local_save_path, local_filename)
    urllib.request.urlretrieve(remote_data_url, full_local_path)
    loaded_file = tarfile.open(full_local_path)
    loaded_file.extractall(path=local_save_path)
    loaded_file.close()


def load_data(path=LOCAL_SAVE_PATH, filename=LOCAL_CSV_FILENAME):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    # not a great solution, since it will break if data order changes or more is added
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    # works, but depends on ids never changing, so new data must always be added to the end
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[-in_test_set], data[in_test_set]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fetch_remote_data()
    raw_data = load_data()
    # print(raw_data.head())
    print(raw_data.info())
    # print(raw_data["ocean_proximity"].value_counts())
    # print(raw_data.describe())
    # raw_data.hist(bins=50, figsize=(20, 15))
    # plt.show()
    train_set, test_set = split_train_test(raw_data, 0.2)
    # print(len(train_set))
    # print(len(test_set))

    # creates new index column to use for train-test split
    raw_data_with_id = raw_data.reset_index()  # adds index column
    # train_set, test_set = split_train_test_by_id(raw_data_with_id, 0.2, "index")
    # print(len(train_set))
    # print(len(test_set))

    # uses lat/long to generate index
    raw_data_with_id["id"] = raw_data["longitude"] * 1000 + raw_data["latitude"] * 1000
    train_set, test_set = split_train_test_by_id(raw_data_with_id, 0.2, "id")
    # consistent, but because lat/long are coarse, there is some overlap of ids:
    # print(len(train_set))
    # print(len(test_set))

    # sklearn has function that splits data randomly:
    train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=42)

    # creating train/test distribution based on income distribution
    raw_data["income_cat"] = pd.cut(raw_data["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
    # raw_data["income_cat"].hist()
    # plt.show()

    # split based on distribution
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(raw_data, raw_data["income_cat"]):
        strat_train_set = raw_data.loc[train_index]
        strat_test_set = raw_data.loc[test_index]

    # print(raw_data["income_cat"].value_counts() / len(raw_data))
    # print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

    # no longer need income_cat column, so remove it from datasets
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    data_copy = strat_train_set.copy()
    # data_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    # plt.show()

    # data_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    #               s=data_copy["population"] / 100, label="population",
    #               figsize=(10, 7), c="median_house_value", cmap=plt.get_cmap("jet"),
    #               colorbar=True, )
    # plt.legend()
    # plt.show()

    # computes _standard correlation coefficient_ between every pair of attributes
    #   a.k.a Pearson's r
    # values close to 1 show strong positive correlation
    # values close to -1 show strong negative correlation
    # values close to 0 show no correlation
    # however, it only detects linear correlation, nonlinear relationships can be missed completely
    corr_matrix = data_copy.corr()
    # display relationship between median_house_value and all others:
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # scatter_matrix() also plots relationships between variables
    # attributes array used to select vars:
    # attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    # scatter_matrix(data_copy[attributes], figsize=(12,8))
    # plt.show()

    # data_copy.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
    # plt.show()

    data_copy["rooms_per_household"] = data_copy["total_rooms"] / data_copy["households"]
    data_copy["bedrooms_per_room"] = data_copy["total_bedrooms"] / data_copy["total_rooms"]
    data_copy["population_per_household"] = data_copy["population"] / data_copy["households"]

    corr_matrix = data_copy.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # separate labels from data because we won't apply teh transforms to the labels
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # data cleaning: what to do in case of missing features
    # can remove rows with bad data:
    # housing.dropna(subset=["total_bedrooms"])
    # can remove column with bad data:
    # housing.drop("total_bedrooms", axis=1)
    # or replace missing data with median:
    # median = housing["total_bedrooms"].median()
    # housing["total_bedrooms"].fillna(median, inplace=True)
    # print(housing.info())

    # or use scikit SimpleImputer to handle missing data:
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)  # drop the text column because imputer only works on numbers
    imputer.fit(housing_num)

    # compare imputer's results with median vals in housing_num:
    print(imputer.statistics_)
    print(housing_num.median().values)

    # now actually replace missing values in training set:
    X = imputer.transform(housing_num)
    # and convert it back to pandas dataframe
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

    # handle text column, because ml training won't handle text+
    housing_cat = housing[["ocean_proximity"]]
    print(housing_cat.head(10))

    # can use scikit to convert text categories to numbers:
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    print(housing_cat_encoded[:10])
    print(ordinal_encoder.categories_)  # array of categories converted

    # can be better to use one-hot encoding:
    from sklearn.preprocessing import OneHotEncoder

    cat_encoder = OneHotEncoder()
    housing_cat_onehot = cat_encoder.fit_transform(housing_cat)
    print(housing_cat_onehot.toarray())
    # and to show the categories:
    print(cat_encoder.categories_)

    # make a custom transformer to add combined columns (e.g. rooms per household)
    from sklearn.base import BaseEstimator, TransformerMixin

    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True):
            self.add_bedrooms_per_room = add_bedrooms_per_room

        def fit(self, X, y=None):
            return self  # do nothing

        def transform(self, X):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]


    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)
    print(housing_extra_attribs)

    # Set up transformation pipelines to perform multiple transforms in sequence
    # StandardScaler applies standardization feature scaling
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    housing_num_tr = num_pipeline.fit_transform(housing_num)

    # combine previous pipeline with text category one hot encoder to handle the whole training dataset
    from sklearn.compose import ColumnTransformer

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])

    housing_prepared = full_pipeline.fit_transform(housing)  # remember housing is training set w/o labels

    ####
    # Do Linear Regression
    ####
    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    some_data = housing.iloc[:5]
    some_labels = housing_labels[:5]
    some_data_prepared = full_pipeline.transform(some_data)

    # see predictions alongside labels:
    print("Predictions:", lin_reg.predict(some_data_prepared))
    print("Labels:", list(some_labels))

    # not so accurate, so view RMSE for the model
    from sklearn.metrics import mean_squared_error

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("Linear regression:", lin_rmse)

    # results not great, they are underfitting the data, so we can:
    #   1) train more data
    #   2) try a more powerful model
    #   3) change/add training features
    #   4) reduce constraints (regularization) on the model

    # try Decision Tree Regressor instead of Linear Regression
    from sklearn.tree import DecisionTreeRegressor

    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)  # this is training the regressor model

    # now do prediction on training data
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print("Decision Tree Regressor:", tree_rmse)

    # try Support Vector Machine regressor
    from sklearn.svm import SVR

    svr_reg = SVR()
    svr_reg.fit(housing_prepared, housing_labels)  # trains the regressor model
    housing_predictions = svr_reg.predict(housing_prepared)
    svr_mse = mean_squared_error(housing_labels, housing_predictions)
    svr_rmse = np.sqrt(svr_mse)
    print("SVR Regressor:", svr_rmse)

    # cross-validation breaks the data into smaller chunks, trains on most of them, then uses last one
    # for evaluation
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)


    def display_scores(dscores):
        print("Scores:", dscores)
        print("Mean:", dscores.mean())
        print("Standard deviation:", dscores.std())


    display_scores(tree_rmse_scores)

    # Decision tree was worse than Linear Regression, so now do cross-validation with linear regression to check:
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)
    # Linear regression still does better. The rmse is slightly smaller

    # now train with Random Forest Regressor. Internally, it trains many Decision Trees on random subset of features
    # then averages their predictions. Using layers of modeis is called _ensemble learning_.
    from sklearn.ensemble import RandomForestRegressor

    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)

    housing_predictions = forest_reg.predict(housing_prepared)
    forest_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_rmse = np.sqrt(forest_mse)
    print("Random Forest Regressor:", forest_rmse)

    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)

    display_scores(forest_rmse_scores)
    # best results so far, but it would still make sense to try more before spending time tweaking hyperparameters
    # on this one

    # Grid Search
    # Grid search will try variations of hyperparameters for you and return the best combination
    from sklearn.model_selection import GridSearchCV

    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)

    print(grid_search.best_params_)
    print(grid_search.best_estimator_)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    # calculate importance of each feature relative to the predictions:
    feature_importances = grid_search.best_estimator_.feature_importances_
    print(feature_importances)

    # display importances along with their column names
    # least important or redundant features might be candidates to drop
    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room", ]
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    print(sorted(zip(feature_importances, attributes), reverse=True))

    # predict against test set
    final_model = grid_search.best_estimator_
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("Final rmse:", final_rmse)

    # compute 95% confidence interval
    from scipy import stats

    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                        loc=squared_errors.mean(),
                                        scale=stats.sem(squared_errors)))
    print("95% confidence interval:", interval)

    """
    param_grid = [
        {'kernel': ["linear"], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000., ]},
        {'kernel': ["rbf"], 'C': [1., 3., 10., 30., 100., 300., 1000., ], 'gamma': [0.01, 0.03, 0.1, 0.3, 1., 3., ]},
    ]

    svr_reg = SVR()
    grid_search = GridSearchCV(svr_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               verbose=2, n_jobs=4)
    grid_search.fit(housing_prepared, housing_labels)
    print("### Grid Search with SVR results ###")
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    """

    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import expon, reciprocal

    params = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }
    svr_reg = SVR()
    random_search = RandomizedSearchCV(svr_reg, param_distributions=params,
                                       n_iter=50, cv=5,
                                       scoring='neg_mean_squared_error',
                                       verbose=2, n_jobs=4, random_state=42)
    random_search.fit(housing_prepared, housing_labels)
    print("### RandomizedSearchCV results ###")
    print(random_search.best_params_)
    print(random_search.best_estimator_)
    cvres = random_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
