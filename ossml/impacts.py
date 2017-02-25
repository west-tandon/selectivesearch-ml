import numpy as np
import pandas as pd
import logging
import math
from sklearn.ensemble import RandomForestRegressor
from ossml.utils import Dataset
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def feature_columns(dataset):
    return [f.name for f in dataset.query_features] + [f.name for f in dataset.shard_features] + ['BID']


def train_payoffs(dataset, n_jobs=-1, n_estimators=20):

    clf = RandomForestRegressor(verbose=True, n_jobs=n_jobs, n_estimators=n_estimators)

    logger.info("Loading dataset")
    training_data = dataset.load()

    features = feature_columns(dataset)
    X = np.array(training_data[features])
    y = np.array(training_data['payoff'])

    cut = math.floor(len(X) * 0.7)
    X, y = shuffle(X, y)
    X_train, y_train = X[:cut], y[:cut]
    X_test, y_test = X[cut:], y[cut:]

    logger.info("Training model")
    clf.fit(X_train, y_train)

    logger.info("Evaluating model")
    y_pred = clf.predict(X_test)
    err = mean_squared_error(y_test, y_pred)
    logger.info("MSE = %f", err)

    logger.info("Feature scores: %s",
                str(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), features), reverse=True)))

    logger.info("Success.")
    return clf, err


def predict_payoffs(dataset, model):
    test_data = dataset.load()
    X = np.array(test_data[feature_columns(dataset)])
    return test_data, pd.DataFrame(model.predict(X))


def run_train(j, out):
    props = Dataset.parse_json(j, 'impact_features')
    features = props['impact_features']
    model, err = train_payoffs(Dataset(features['query'], features['shard'], features['bucket'], props['buckets']))
    joblib.dump(model, out)


def run_predict(j, model_path):
    props = Dataset.parse_json(j, 'impact_features')
    features = props['impact_features']
    logger.info("Loading model")
    model = joblib.load(model_path)
    logger.info("Loading dataset")
    dataset = Dataset(features['query'], features['shard'], features['bucket'], props['buckets'])
    logger.info("Making predictions")
    X, y = predict_payoffs(dataset, model)
    X['payoff'] = y
    basename = props['basename']
    logger.info("Storing predictions")
    for shard, shard_group in X.groupby('SID'):
        for bucket, bucket_group in shard_group.groupby('BID'):
            with open("{0}#{1}#{2}.payoff".format(basename, shard, bucket), 'w') as f:
                for idx, x in bucket_group.sort_values(by='QID').iterrows():
                    f.write(str(x['payoff']) + "\n")
    logger.info("Success.")
