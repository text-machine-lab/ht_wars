"""David Donahue 2016. Use XGBoost to create a boosted decision tree over tweet pair features."""
import abc

import numpy as np

import lightgbm as lgb
import xgboost as xgb

from sacred.observers import MongoObserver
from sacred import Experiment

from tools import get_hashtag_file_names
from config import SEMEVAL_HUMOR_TRAIN_DIR
from config import SEMEVAL_HUMOR_TRIAL_DIR
from config import BOOST_TREE_TWEET_PAIR_TRAINING_DIR
from config import BOOST_TREE_TWEET_PAIR_TESTING_DIR
from config import MONGO_ADDRESS

ex_name = 'hashtagwars_boost_tree'
ex = Experiment(ex_name)

ex.observers.append(MongoObserver.create(url=MONGO_ADDRESS, db_name=ex_name))


class BaseTreeModel(object):
    def __init__(self, **kwargs):
        super(BaseTreeModel, self).__init__()

        self.model = None
        self.param = kwargs

        self.zzz = None

    def prepare_data(self, data, labels):
        """Converts the data to the format required by the specific model"""
        return data, labels

    @abc.abstractmethod
    def train(self, data, labels, data_val=None):
        """Trains the model"""

    @abc.abstractmethod
    def predict(self, data):
        """Returns predictions on the test data"""

    def evaluate(self, data, labels):
        """Retunrs the accuracy on the given data"""

        np_preds = self.predict(data)
        np_preds_classified = np.round(np_preds)
        accuracy = np.mean(labels == np_preds_classified)

        return accuracy


class XGBoostTreeModel(BaseTreeModel):
    def __init__(self, **kwargs):
        super(XGBoostTreeModel, self).__init__(**kwargs)

    def prepare_data(self, data, labels):
        if not isinstance(data, xgb.DMatrix):
            data = xgb.DMatrix(data, label=labels)

        return data, labels

    def train(self, data, labels, data_val=None):
        data, _ = self.prepare_data(data, labels)

        num_round = self.param.pop('num_round')
        self.model = xgb.train(self.param, data, num_round)

    def predict(self, data):
        data, _ = self.prepare_data(data, None)

        predicted = self.model.predict(data)
        return predicted


class LightGBMTreeModel(BaseTreeModel):
    def __init__(self, **kwargs):
        super(LightGBMTreeModel, self).__init__(**kwargs)

    def prepare_data(self, data, labels):
        if not isinstance(data, lgb.Dataset):
            data = lgb.Dataset(data, label=labels, free_raw_data=False)

        return data, labels

    def train(self, data, labels, data_val=None):
        data, _ = self.prepare_data(data, labels)

        num_round = self.param.pop('num_round')
        self.model = lgb.train(self.param, data, num_round)

    def predict(self, data):
        data, _ = self.prepare_data(data, None)

        predicted = self.model.predict(data.data)  # LightGBM's `predict` requires raw data
        return predicted


@ex.config
def config():
    tree_model_lib = ''


@ex.named_config
def xgboost():
    tree_model_lib = 'xgboost'

    objective = 'binary:logistic'

    max_depth = 4
    eta = 0.020  # learning_rate
    gamma = 4  # min_split_loss
    reg_lambda = 2.5e-06
    num_round = 19
    silent = 0


@ex.named_config
def lightgbm():
    tree_model_lib = 'lightgbm'

    objective = 'binary'

    num_round = 10
    learning_rate = 0.1
    num_leaves = 127
    lambda_l2 = 0.005
    max_bin = 255
    min_gain_to_split = 0.0


@ex.main
def main(_config):
    tree_model_lib = _config.pop('tree_model_lib')
    print 'Starting program using', tree_model_lib

    train_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRAIN_DIR)
    list_of_labels = []
    list_of_datas = []
    for hashtag_name in train_hashtag_names:
        np_hashtag_labels = np.load(open(BOOST_TREE_TWEET_PAIR_TRAINING_DIR + hashtag_name + '_labels.npy', 'rb'))
        np_hashtag_data = np.load(open(BOOST_TREE_TWEET_PAIR_TRAINING_DIR + hashtag_name + '_data.npy', 'rb'))
        list_of_labels.append(np_hashtag_labels)
        list_of_datas.append(np_hashtag_data)
    np_data = np.vstack(list_of_datas)
    np_labels = np.concatenate(list_of_labels, axis=0)
    print 'Data:', np_data.shape
    print 'Labels', np_labels.shape

    # create and train the model

    model = None
    if tree_model_lib == 'xgboost':
        model = XGBoostTreeModel(**_config)
    elif tree_model_lib == 'lightgbm':
        model = LightGBMTreeModel(**_config)
    else:
        raise AttributeError('Model is not known: {}'.format(tree_model_lib))

    data_converted, labels_converted = model.prepare_data(np_data, np_labels)
    model.train(data_converted, labels_converted)

    accuracy_train = model.evaluate(data_converted, labels_converted)
    print 'Accuracy train:', accuracy_train

    # test on the test data
    test_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRIAL_DIR)
    list_of_accuracies = []
    for hashtag_name in test_hashtag_names:
        print 'Testing on hashtag %s' % hashtag_name
        np_label_test = np.load(open(BOOST_TREE_TWEET_PAIR_TESTING_DIR + hashtag_name + '_labels.npy', 'rb'))
        np_data_test = np.load(open(BOOST_TREE_TWEET_PAIR_TESTING_DIR + hashtag_name + '_data.npy', 'rb'))

        accuracy_test = model.evaluate(np_data_test, np_label_test)
        list_of_accuracies.append(accuracy_test)
        print 'Hashtag test accuracy: %s' % accuracy_test

    accuracy_test_mean = np.mean(list_of_accuracies)
    print 'Mean test accuracy: %s' % accuracy_test_mean

    result = {
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test_mean,
    }

    return result


if __name__ == '__main__':
    ex.run_commandline()
