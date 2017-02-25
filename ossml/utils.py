import abc
import copy
import re

import pandas as pd


class Feature(metaclass=abc.ABCMeta):

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self._data = None

    def data(self):
        if self._data is None:
            self.load_data()
        return self._data

    @abc.abstractmethod
    def load_data(self):
        """Load the data to the attribute _data"""

    def __str__(self):
        return "[" + self.name + "," + self.path + "]"

    def __repr__(self):
        return self.__str__()


@Feature.register
class QueryFeature(Feature):

    def load_data(self):
        self._data = pd.read_csv("{0}.{1}".format(self.path, self.name), header=None, names=[self.name])
        self._data['QID'] = self._data.index
        self._data = self._data[['QID', self.name]]


@Feature.register
class ShardFeature(Feature):

    def __init__(self, name, path, num_shards):
        super().__init__(name, path)
        self.num_shards = num_shards
        pass

    def shard_df(self, shard):
        df = pd.read_csv("{0}#{1}.{2}".format(self.path, shard, self.name), header=None, names=[self.name])
        df['QID'] = df.index
        df['SID'] = pd.Series(shard, index=df.index)
        return df

    def load_data(self):
        shards = [self.shard_df(shard) for shard in range(self.num_shards)]
        self._data = pd.concat(shards, ignore_index=True, copy=False)
        self._data = self._data[['QID', 'SID', self.name]]

    def __str__(self):
        return "[" + self.name + "," + self.path + "," + str(self.num_shards) + "]"


@Feature.register
class BucketFeature(Feature):

    def __init__(self, name, path, num_shards, num_buckets):
        super().__init__(name, path)
        self.num_shards = num_shards
        self.num_buckets = num_buckets
        pass

    def bucket_df(self, shard, bucket):
        df = pd.read_csv("{0}#{1}#{2}.{3}".format(self.path, shard, bucket, self.name), header=None, names=[self.name])
        df['QID'] = df.index
        df['SID'] = pd.Series(shard, index=df.index)
        df['BID'] = pd.Series(bucket, index=df.index)
        return df

    def shard_df(self, shard):
        buckets = [self.bucket_df(shard, bucket) for bucket in range(self.num_buckets)]
        shard = pd.concat(buckets, ignore_index=True, copy=False)
        return shard[['QID', 'SID', 'BID', self.name]]

    def load_data(self):
        shards = [self.shard_df(shard) for shard in range(self.num_shards)]
        self._data = pd.concat(shards, ignore_index=True, copy=False)
        self._data = self._data[['QID', 'SID', 'BID', self.name]]

    def __str__(self):
        return "[" + self.name + "," + self.path + "," + str(self.num_shards) + "," + str(self.num_buckets) + "]"


class Dataset:

    @staticmethod
    def resolve_reference(j, path):
        for elem in path.split("/")[1:]:
            j = j[elem]
        return j

    @staticmethod
    def parse_path(j, path, features_field):
        i = path.find(':')
        if i > -1:
            p = path[:i]
            pattern = re.compile('\$\{([^\}]*)\}')
            matches = list(pattern.finditer(p))
            end = 0
            s = []
            for match in matches:
                s.append(p[end:match.span()[0]])
                s.append(Dataset.resolve_reference(j, match.group(1)))
                end = match.span()[1]
            s.append(p[end:])
            p = ''.join(s)
        else:
            p = j[features_field]['base']
        return path[i + 1:], p

    @staticmethod
    def parse_json(j, features_field='features'):
        """Resolves paths to features"""
        parsed = copy.deepcopy(j)
        num_shards = parsed['shards']
        num_buckets = parsed['buckets'] if 'buckets' in parsed else None
        parsed[features_field]['query'] = [QueryFeature(*Dataset.parse_path(parsed, path, features_field))
                                           for path in parsed[features_field]['query']]
        parsed[features_field]['shard'] = [ShardFeature(*Dataset.parse_path(parsed, path, features_field), num_shards)
                                           for path in parsed[features_field]['shard']]
        if 'bucket' in parsed[features_field]:
            parsed[features_field]['bucket'] =\
                [BucketFeature(*Dataset.parse_path(parsed, path, features_field), num_shards, num_buckets)
                 for path in parsed[features_field]['bucket']]
        else:
            parsed[features_field]['bucket'] = None
        return parsed

    def __init__(self, query_features, shard_features, bucket_features, num_buckets):
        self.query_features = query_features
        self.shard_features = shard_features
        self.bucket_features = bucket_features
        self.num_buckets = num_buckets

    @staticmethod
    def merge(features, on):
        assert len(features) > 0
        merged = features[0]
        for feature in features[1:]:
            merged = pd.merge(merged, feature, on=on)
        return merged

    def load(self):
        assert self.query_features is not None
        assert len(self.query_features) > 0
        assert self.shard_features is not None
        assert len(self.shard_features) > 0
        qf = self.merge([f.data() for f in self.query_features], on='QID')
        sf = self.merge([f.data() for f in self.shard_features], on=['QID', 'SID'])
        dataset = pd.merge(qf, sf, on=['QID'])
        if self.bucket_features is not None:
            if len(self.bucket_features) > 0:
                bf = self.merge([f.data() for f in self.bucket_features], on=['QID', 'SID', 'BID'])
                dataset = pd.merge(dataset, bf, on=['QID', 'SID'])
            else:
                bf = pd.DataFrame({
                    'BID': range(self.num_buckets),
                    'key': 1
                })
                columns = list(dataset.columns.values) + ['BID']
                dataset['key'] = 1
                dataset = pd.merge(dataset, bf, on=['key'])[columns]
        return dataset

