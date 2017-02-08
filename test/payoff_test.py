import os
import shutil
import tempfile
from sklearn.externals import joblib
from unittest.mock import MagicMock

import pandas as pd

import ossml.payoffs as pf
from ossml.utils import BucketFeature
from ossml.utils import Dataset
from test.utils_test import UtilsTest


class PayoffsTest(UtilsTest):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def feature_path(self):
        return os.path.join(self.test_dir, "path")

    def test_train_payoff(self):
        # given
        dataset = Dataset([self.qf1(), self.qf2()],
                          [self.sf1(), self.sf2()],
                          [BucketFeature("payoff", '', 2, 2)], 2)
        df = pd.DataFrame({
            'QID': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            'qf1': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            'qf2': [10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30],
            'SID': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            'sf1': [1, 1, 11, 11, 2, 2, 22, 22, 3, 3, 33, 33],
            'sf2': [10, 10, 110, 110, 20, 20, 220, 220, 30, 30, 330, 330],
            'BID': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'payoff': [1, 4, 11, 44, 2, 5, 22, 55, 3, 6, 33, 66]
        })[['QID', 'qf1', 'qf2', 'SID', 'sf1', 'sf2', 'BID', 'payoff']]
        dataset.load = MagicMock(return_value=df)
        model, err = pf.train_payoffs(dataset)

        dataset.load = MagicMock(return_value=df[['QID', 'qf1', 'qf2', 'SID', 'sf1', 'sf2', 'BID']])
        predicted = pf.predict_payoffs(dataset, model)

    def test_run_train_and_run_predict(self):
        j = {
            "basename": os.path.join(self.test_dir, "basename"),
            "shards": 2,
            "buckets": 2,
            "features": {
                "base": self.feature_path(),
                "query": [f.name for f in [self.qf1(), self.qf2()]],
                "shard": [f.name for f in [self.sf1(), self.sf2()]],
                "bucket": [self.bf1().name]
            }
        }
        model_path = os.path.join(self.test_dir, "model")
        pf.run_train(j, model_path)

        j['features']['bucket'] = []
        pf.run_predict(j, model_path)

        for shard in range(2):
            for bucket in range(2):
                with open("{0}#{1}#{2}.payoff".format(j['basename'], shard, bucket)) as f:
                    lines = f.readlines()
                    self.assertEqual(len(lines), 3)
                    for v in lines:
                        float(v)
