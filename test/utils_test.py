import os
import shutil
import tempfile
import unittest
import json

import pandas as pd
from pandas.util.testing import assert_frame_equal

from ossml.utils import BucketFeature
from ossml.utils import Dataset
from ossml.utils import QueryFeature
from ossml.utils import ShardFeature


class UtilsTest(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def feature_path(self):
        return "path"

    def qf1(self):
        feature = QueryFeature("qf1", os.path.join(self.test_dir, self.feature_path()))
        with open("{0}.{1}".format(feature.path, feature.name), 'w') as f:
            f.write('1\n2\n3\n')
        return feature

    def qf2(self):
        feature = QueryFeature("qf2", os.path.join(self.test_dir, self.feature_path()))
        with open("{0}.{1}".format(feature.path, feature.name), 'w') as f:
            f.write('10\n20\n30\n')
        return feature

    def sf1(self):
        feature = ShardFeature("sf1", os.path.join(self.test_dir, self.feature_path()), 2)
        with open("{0}#{1}.{2}".format(feature.path, 0, feature.name), 'w') as f:
            f.write('1\n2\n3\n')
        with open("{0}#{1}.{2}".format(feature.path, 1, feature.name), 'w') as f:
            f.write('11\n22\n33\n')
        return feature

    def sf2(self):
        feature = ShardFeature("sf2", os.path.join(self.test_dir, self.feature_path()), 2)
        with open("{0}#{1}.{2}".format(feature.path, 0, feature.name), 'w') as f:
            f.write('10\n20\n30\n')
        with open("{0}#{1}.{2}".format(feature.path, 1, feature.name), 'w') as f:
            f.write('110\n220\n330\n')
        return feature

    def bf1(self):
        feature = BucketFeature("payoff", os.path.join(self.test_dir, self.feature_path()), 2, 2)
        with open("{0}#{1}#{2}.{3}".format(feature.path, 0, 0, feature.name), 'w') as f:
            f.write('1\n2\n3\n')
        with open("{0}#{1}#{2}.{3}".format(feature.path, 0, 1, feature.name), 'w') as f:
            f.write('4\n5\n6\n')
        with open("{0}#{1}#{2}.{3}".format(feature.path, 1, 0, feature.name), 'w') as f:
            f.write('11\n22\n33\n')
        with open("{0}#{1}#{2}.{3}".format(feature.path, 1, 1, feature.name), 'w') as f:
            f.write('44\n55\n66\n')
        return feature

    def test_query_feature(self):
        # given
        feature = self.qf1()
        # when
        data = feature.data()
        # then
        assert_frame_equal(data, pd.DataFrame({'QID': [0, 1, 2], feature.name: [1, 2, 3]}))

    def test_shard_feature(self):
        # given
        feature = self.sf1()
        # when
        data = feature.data()
        # then
        assert_frame_equal(data, pd.DataFrame({
            'QID': [0, 1, 2, 0, 1, 2],
            'SID': [0, 0, 0, 1, 1, 1],
            feature.name: [1, 2, 3, 11, 22, 33]
        }))

    def test_bucket_feature(self):
        # given
        feature = self.bf1()
        # when
        data = feature.data()
        # then
        assert_frame_equal(data, pd.DataFrame({
            'QID': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            'SID': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            'BID': [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            feature.name: [1, 2, 3, 4, 5, 6, 11, 22, 33, 44, 55, 66]
        })[['QID', 'SID', 'BID', feature.name]])

    def test_merge_qf(self):
        # given
        f1 = self.qf1()
        f2 = self.qf2()
        # when
        data = Dataset.merge([f1.data(), f2.data()], 'QID')
        # then
        assert_frame_equal(data, pd.DataFrame({
            'QID': [0, 1, 2],
            f1.name: [1, 2, 3],
            f2.name: [10, 20, 30]
        }))

    def test_merge_sf(self):
        # given
        f1 = self.sf1()
        f2 = self.sf2()
        # when
        data = Dataset.merge([f1.data(), f2.data()], ['QID', 'SID'])
        # then
        assert_frame_equal(data, pd.DataFrame({
            'QID': [0, 1, 2, 0, 1, 2],
            'SID': [0, 0, 0, 1, 1, 1],
            f1.name: [1, 2, 3, 11, 22, 33],
            f2.name: [10, 20, 30, 110, 220, 330]
        }))

    def test_load_dataset(self):
        assert_frame_equal(
            Dataset([self.qf1(), self.qf2()],
                    [self.sf1(), self.sf2()],
                    [self.bf1()], 2).load(),
            pd.DataFrame({
                'QID': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                'qf1': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                'qf2': [10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30],
                'SID': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                'sf1': [1, 1, 11, 11, 2, 2, 22, 22, 3, 3, 33, 33],
                'sf2': [10, 10, 110, 110, 20, 20, 220, 220, 30, 30, 330, 330],
                'BID': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                'payoff': [1, 4, 11, 44, 2, 5, 22, 55, 3, 6, 33, 66]
            })[['QID', 'qf1', 'qf2', 'SID', 'sf1', 'sf2', 'BID', 'payoff']]
        )

    def test_load_dataset_with_empty_bf(self):
        assert_frame_equal(
            Dataset([self.qf1(), self.qf2()],
                    [self.sf1(), self.sf2()],
                    [], 2).load(),
            pd.DataFrame({
                'QID': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                'qf1': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                'qf2': [10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30],
                'SID': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                'sf1': [1, 1, 11, 11, 2, 2, 22, 22, 3, 3, 33, 33],
                'sf2': [10, 10, 110, 110, 20, 20, 220, 220, 30, 30, 330, 330],
                'BID': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            })[['QID', 'qf1', 'qf2', 'SID', 'sf1', 'sf2', 'BID']]
        )

    def test_parse_json(self):
        j = json.loads('''{
          "basename": "a",
          "shards": 2,
          "buckets": 2,
          "features": {
            "base": "b",
            "query": ["${/basename}${/features/base}:qf1", "qf2"],
            "shard": ["/c:sf1", "sf2"],
            "bucket": ["${/basename}:bf1"]
          }
        }''')
        actual = Dataset.parse_json(j)
        expected = {
          "basename": "a",
          "shards": 2,
          "buckets": 2,
          "features": {
            "base": "b",
            "query": [QueryFeature("qf1", "ab"), QueryFeature("qf2", "b")],
            "shard": [ShardFeature("sf1", "/c", 2), ShardFeature("sf2", "b", 2)],
            "bucket": [BucketFeature("bf1", "a", 2, 2)]
          }
        }
        self.assertEqual(str(actual), str(expected))

    def test_parse_json_no_bucket(self):
        j = json.loads('''{
          "basename": "a",
          "shards": 2,
          "buckets": 2,
          "features": {
            "base": "b",
            "query": ["${/basename}${/features/base}:qf1", "qf2"],
            "shard": ["/c:sf1", "sf2"]
          }
        }''')
        actual = Dataset.parse_json(j)
        expected = {
          "basename": "a",
          "shards": 2,
          "buckets": 2,
          "features": {
            "base": "b",
            "query": [QueryFeature("qf1", "ab"), QueryFeature("qf2", "b")],
            "shard": [ShardFeature("sf1", "/c", 2), ShardFeature("sf2", "b", 2)],
            "bucket": None
          }
        }
        self.assertEqual(str(actual), str(expected))