"""David Donahue 2016. Use XGBoost to create a boosted decision tree over tweet pair features."""
import abc
import cPickle as pickle

import numpy as np

import lightgbm as lgb
import xgboost as xgb

from sacred.observers import MongoObserver
from sacred import Experiment

from tools import get_hashtag_file_names
from config import SEMEVAL_HUMOR_TRAIN_DIR, BOOST_TREE_MODEL_FILE_PATH, BOOST_TREE_EVAL_TWEET_PAIR_PREDICTIONS, \
    SEMEVAL_HUMOR_EVAL_DIR, BOOST_TREE_TWEET_PAIR_EVAL_DIR, BOOST_TREE_TRIAL_TWEET_PAIR_PREDICTIONS, \
    HUMOR_EVAL_PREDICTION_HASHTAGS
from config import SEMEVAL_HUMOR_TRIAL_DIR
from config import BOOST_TREE_TWEET_PAIR_TRAIN_DIR
from config import BOOST_TREE_TWEET_PAIR_TRIAL_DIR
from config import MONGO_ADDRESS

ex_name = 'hashtagwars_boost_tree'
ex = Experiment(ex_name)

# ex.observers.append(MongoObserver.create(url=MONGO_ADDRESS, db_name=ex_name))


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

    @abc.abstractmethod
    def save_model(self, filename):
        """Saves the trained model into a file"""

    @abc.abstractmethod
    def restore_model(self, filename):
        """Restores the trained model from a file"""


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

    def save_model(self, filename):
        self.model.save_model(filename)

    def restore_model(self, filename):
        self.model = xgb.Booster({'nthread': 4})
        self.model.load_model(filename)


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




def load_tree_data(hashtags, base_dir):
    if not isinstance(hashtags, list):
        hashtags = [hashtags]

    list_of_labels = []
    list_of_datas = []
    for hashtag_name in hashtags:
        np_hashtag_data = np.load(open(base_dir + hashtag_name + '_data.npy', 'rb'))
        list_of_datas.append(np_hashtag_data)

        np_hashtag_labels = np.load(open(base_dir + hashtag_name + '_labels.npy', 'rb'))
        list_of_labels.append(np_hashtag_labels)

    data_train = np.vstack(list_of_datas)
    labels_train = np.concatenate(list_of_labels, axis=0)

    return data_train, labels_train



@ex.config
def config():
    tree_model_lib = ''


@ex.named_config
def xgboost():
    tree_model_lib = 'xgboost'

    objective = 'binary:logistic'

    max_depth = 4
    eta = 0.038  # learning_rate
    gamma = 2  # min_split_loss
    reg_lambda = 1.03e-05
    num_round = 10
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
    print 'Hashtags:', len(train_hashtag_names)

    np_data, np_labels = load_tree_data(train_hashtag_names, BOOST_TREE_TWEET_PAIR_TRAIN_DIR)
    print 'Data:', np_data.shape, np_labels.shape

    train_hashtag_names_trial = get_hashtag_file_names(SEMEVAL_HUMOR_TRIAL_DIR)
    print 'Hashtags trial:', len(train_hashtag_names_trial)

    np_data_trial, np_labels_trial = load_tree_data(train_hashtag_names_trial, BOOST_TREE_TWEET_PAIR_TRIAL_DIR)
    print 'Data trial:', np_data_trial.shape, np_labels_trial.shape

    # create and train the model
    if tree_model_lib == 'xgboost':
        model = XGBoostTreeModel(**_config)
    elif tree_model_lib == 'lightgbm':
        model = LightGBMTreeModel(**_config)
    else:
        raise AttributeError('Model is not known: {}'.format(tree_model_lib))

    np_data_combined = np.concatenate([np_data, np_data_trial], axis=0)
    np_labels_combined = np.concatenate([np_labels, np_labels_trial], axis=0)
    print 'Data trial:', np_data_combined.shape, np_labels_combined.shape

    data_converted, labels_converted = model.prepare_data(np_data, np_labels)
    model.train(data_converted, labels_converted)

    accuracy_train = model.evaluate(data_converted, labels_converted)
    print 'Accuracy train:', accuracy_train

    model.save_model(BOOST_TREE_MODEL_FILE_PATH)

    result = {
        'accuracy_train': accuracy_train,
    }

    return result


@ex.command
def predict(_config):
    tree_model_lib = _config.pop('tree_model_lib')
    print 'Starting program using', tree_model_lib

    hashtag_filenames = HUMOR_EVAL_PREDICTION_HASHTAGS
    tree_data_dir = BOOST_TREE_TWEET_PAIR_EVAL_DIR
    output_dir = BOOST_TREE_EVAL_TWEET_PAIR_PREDICTIONS

    # create and train the model
    if tree_model_lib == 'xgboost':
        model = XGBoostTreeModel(**_config)
    elif tree_model_lib == 'lightgbm':
        model = LightGBMTreeModel(**_config)
    else:
        raise AttributeError('Model is not known: {}'.format(tree_model_lib))

    model.restore_model(BOOST_TREE_MODEL_FILE_PATH)

    # predict on each hashtag
    # predict_hashtag_names = get_hashtag_file_names(source_dir)
    with open(hashtag_filenames, 'rb') as f:
        hashtag_names = pickle.load(f)

    print 'Hashtags:', len(hashtag_names)

    hashtag_predictions = []
    for hashtag_name in hashtag_names:
        np_data, np_labels = load_tree_data(hashtag_name, tree_data_dir)

        predictions = model.predict(np_data)
        hashtag_predictions.append(predictions)

    # save the predictions
    with open(output_dir, 'wb') as f:
        pickle.dump(hashtag_predictions, f)

    print 'Predictions saved:', output_dir


if __name__ == '__main__':
    ex.run_commandline()
