"""David Donahue 2016. Use XGBoost to create a boosted decision tree over tweet pair features."""
import xgboost as xgb
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

from tools import get_hashtag_file_names
from config import SEMEVAL_HUMOR_TRAIN_DIR
from config import SEMEVAL_HUMOR_TRIAL_DIR
from config import BOOST_TREE_TWEET_PAIR_TRAINING_DIR
from config import BOOST_TREE_TWEET_PAIR_TESTING_DIR
from config import MONGO_ADDRESS

ex_name = 'hashtagwars_boost_tree'
ex = Experiment(ex_name)

ex.observers.append(MongoObserver.create(url=MONGO_ADDRESS, db_name=ex_name))

@ex.config
def config():
    max_depth = 12
    eta = 0.003  # learning_rate
    gamma = 3  # min_split_loss
    reg_lambda = 0.007
    num_round = 19
    silent = 0


@ex.main
def main(num_round, max_depth, eta, gamma, reg_lambda, silent):
    print 'Starting program'
    param = {
        'silent': silent,
        'objective': 'binary:logistic',

        'max_depth': max_depth,
        'eta': eta,
        'gamma': gamma,
        'reg_lambda': reg_lambda
    }

    train_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRAIN_DIR)
    bst = None
    list_of_labels = []
    list_of_datas = []
    for hashtag_name in train_hashtag_names:
        print 'Training on hashtag %s' % hashtag_name
        np_hashtag_labels = np.load(open(BOOST_TREE_TWEET_PAIR_TRAINING_DIR + hashtag_name + '_labels.npy', 'rb'))
        print np_hashtag_labels.shape
        np_hashtag_data = np.load(open(BOOST_TREE_TWEET_PAIR_TRAINING_DIR + hashtag_name + '_data.npy', 'rb'))
        list_of_labels.append(np_hashtag_labels)
        list_of_datas.append(np_hashtag_data)
    np_data = np.vstack(list_of_datas)
    np_labels = np.concatenate(list_of_labels, axis=0)
    print 'Data:', np_data.shape
    print 'Labels', np_labels.shape

    dtrain = xgb.DMatrix(np_data, label=np_labels)
    bst = xgb.train(param, dtrain, num_round)
    np_preds = bst.predict(dtrain)
    np_preds_classified = np.round(np_preds)
    accuracy_train = np.mean(np_labels == np_preds_classified)
    print 'Accuracy:', accuracy_train

    test_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRIAL_DIR)
    list_of_accuracies = []
    for hashtag_name in test_hashtag_names:
        print 'Testing on hashtag %s' % hashtag_name
        np_label = np.load(open(BOOST_TREE_TWEET_PAIR_TESTING_DIR + hashtag_name + '_labels.npy', 'rb'))
        np_data = np.load(open(BOOST_TREE_TWEET_PAIR_TESTING_DIR + hashtag_name + '_data.npy', 'rb'))
        dtest = xgb.DMatrix(np_data, label=np_label)
        np_preds = bst.predict(dtest)
        np_preds_classified = np.round(np_preds)
        accuracy = np.mean(np_label == np_preds_classified)
        list_of_accuracies.append(accuracy)
        print 'Hashtag test accuracy: %s' % accuracy

    accuracy_test_mean = np.mean(list_of_accuracies)
    print 'Mean test accuracy: %s' % accuracy_test_mean

    result = {
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test_mean,
    }

    return result

if __name__ == '__main__':
    ex.run_commandline()
