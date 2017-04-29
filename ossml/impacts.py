import numpy as np
import pandas as pd
import logging
import math
import fastparquet

from fastparquet import write
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_train(j, out):
    features = j['impact_features']
    basename = j['basename']
    features_basename = features['basename']

    logger.info("Loading data")

    index = ['query', 'shard']

    taily_features = fastparquet.ParquetFile('{}.taily'.format(features_basename))\
        .to_pandas(columns=index + features['taily'])
    redde_features = fastparquet.ParquetFile('{}.redde'.format(features_basename))\
        .to_pandas(columns=index + features['redde'])
    ranks_features = fastparquet.ParquetFile('{}.ranks'.format(features_basename))\
        .to_pandas(columns=index + features['ranks'])
    impacts = pd.concat([fastparquet.ParquetFile('{}#{}.impacts'.format(basename, shard)).to_pandas()
                         for shard in range(j['shards'])])
    bucket_ranks = fastparquet.ParquetFile('{}.bucketrank-{}'.format(features_basename, j['buckets'])).to_pandas()

    logger.info("Joining data")

    data = pd.merge(
        pd.merge(
            pd.merge(taily_features, redde_features, on=index),
            ranks_features,
            on=index
        ),
        impacts,
        on=index
    )
    data = pd.merge(data, bucket_ranks, on=['shard', 'bucket'])

    logger.info("Pre-processing data")

    clf = RandomForestRegressor(verbose=True, n_jobs=-1, n_estimators=20)
    feature_names = features['taily'] + features['redde'] + features['ranks'] + ['bucketrank']
    features = np.array(data[feature_names])
    labels = np.array(data['impact'])

    cut = math.floor(len(features) * 0.7)
    features, labels = shuffle(features, labels)
    features_train, labels_train = features[:cut], labels[:cut]
    features_test, labels_test = features[cut:], labels[cut:]

    logger.info("Training model")
    clf.fit(features_train, labels_train)

    logger.info("Evaluating model")
    labels_pred = clf.predict(features_test)
    err = mean_squared_error(labels_test, labels_pred)
    logger.info("MSE = %f", err)

    logger.info("Feature scores: %s",
                str(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), feature_names), reverse=True)))

    logger.info("Success.")
    joblib.dump(clf, out)


def run_predict(j, model_path):
    features = j['impact_features']
    basename = j['basename']
    features_basename = features['basename']

    logger.info("Loading data")

    index = ['query', 'shard']

    taily_features = fastparquet.ParquetFile('{}.taily'.format(features_basename)) \
        .to_pandas(columns=index + features['taily'])
    redde_features = fastparquet.ParquetFile('{}.redde'.format(features_basename)) \
        .to_pandas(columns=index + features['redde'])
    ranks_features = fastparquet.ParquetFile('{}.ranks'.format(features_basename)) \
        .to_pandas(columns=index + features['ranks'])
    bucket_ranks = fastparquet.ParquetFile('{}.bucketrank-{}'.format(features_basename, j['buckets'])).to_pandas()

    logger.info("Joining data")

    features_data = pd.merge(
        pd.merge(taily_features, redde_features, on=index),
        ranks_features,
        on=index
    )
    data = pd.merge(features_data, bucket_ranks, on=['shard'])
    feature_names = features['taily'] + features['redde'] + features['ranks'] + ['bucketrank']

    model = joblib.load(model_path)
    predictions = pd.DataFrame(model.predict(np.array(data[feature_names])))
    data['impact'] = predictions

    logger.info("Storing predictions")
    for shard, shard_group in data.groupby('shard'):
        write('{}#{}.impacts'.format(basename, shard),
              shard_group[['query', 'shard', 'bucket', 'impact']].sort_values(by=['query', 'bucket']),
              compression='SNAPPY',
              write_index=False)

    logger.info("Success.")