import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ossml.utils import Dataset
from sklearn.externals import joblib


def feature_columns(dataset):
    return [f.name for f in dataset.query_features] + [f.name for f in dataset.shard_features] + ['BID']


def train_payoffs(dataset, n_jobs=-1):
    clf = RandomForestRegressor(verbose=True, n_jobs=n_jobs)
    training_data = dataset.load()
    X = np.array(training_data[feature_columns(dataset)])
    y = np.array(training_data['payoff'])
    clf.fit(X, y)
    return clf


def predict_payoffs(dataset, model):
    test_data = dataset.load()
    X = np.array(test_data[feature_columns(dataset)])
    return test_data, pd.DataFrame(model.predict(X))


def run_train(j, out):
    props = Dataset.parse_json(j)
    features = props['features']
    model = train_payoffs(Dataset(features['query'], features['shard'], features['bucket'], props['buckets']))
    joblib.dump(model, out)


def run_predict(j, model_path):
    props = Dataset.parse_json(j)
    features = props['features']
    model = joblib.load(model_path)
    dataset = Dataset(features['query'], features['shard'], features['bucket'], props['buckets'])
    X, y = predict_payoffs(dataset, model)
    X['payoff'] = y
    basename = props['basename']
    for shard, shard_group in X.groupby('SID'):
        for bucket, bucket_group in shard_group.groupby('BID'):
            with open("{0}#{1}#{2}.payoff".format(basename, shard, bucket), 'w') as f:
                for idx, x in bucket_group.sort_values(by='QID').iterrows():
                    f.write(str(x['payoff']) + "\n")
