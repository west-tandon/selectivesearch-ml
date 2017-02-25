import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pandas as pd

import ossml.costs as costs
from ossml.utils import ShardFeature
from ossml.utils import Dataset
from test.utils_test import UtilsTest


class CostTest(UtilsTest):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def feature_path(self):
        return os.path.join(self.test_dir, "path")

    def test_train_cost(self):
        # given
        dataset = Dataset([self.qf1(), self.qf2()],
                          [self.sf1(), ShardFeature("cost", '', 2)],
                          None, 2)
        df = pd.DataFrame({
            'QID': [0, 0, 1, 1, 2, 2],
            'qf1': [1, 1, 2, 2, 3, 3],
            'qf2': [10, 10, 20, 20, 30, 30],
            'SID': [0, 1, 0, 1, 0, 1],
            'sf1': [1, 11, 2, 22, 3, 33],
            'cost': [10, 110, 20, 220, 30, 330],
        })[['QID', 'qf1', 'qf2', 'SID', 'sf1', 'cost']]
        dataset.load = MagicMock(return_value=df)
        model, err = costs.train_costs(dataset)

        dataset.load = MagicMock(return_value=df[['QID', 'qf1', 'qf2', 'SID', 'sf1']])
        costs.predict_costs(dataset, model)

    def test_run_train_and_run_predict(self):
        j = {
            "basename": os.path.join(self.test_dir, "basename"),
            "shards": 2,
            "buckets": 2,
            "cost_features": {
                "base": self.feature_path(),
                "query": [f.name for f in [self.qf1(), self.qf2()]],
                "shard": [f.name for f in [self.sf1(), self.cost()]]
            }
        }
        model_path = os.path.join(self.test_dir, "model")
        costs.run_train(j, model_path)

        costs.run_predict(j, model_path)

        for shard in range(2):
            with open("{0}#{1}.cost".format(j['cost_features']['base'], shard)) as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 3)
                for v in lines:
                    float(v)
