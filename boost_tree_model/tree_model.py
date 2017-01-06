"""David Donahue 2016. Use XGBoost to create a boosted decision tree over tweet pair features."""
import xgboost as xgb
import numpy as np
from tools import get_hashtag_file_names
from config import SEMEVAL_HUMOR_TRAIN_DIR
from config import SEMEVAL_HUMOR_TRIAL_DIR
from config import BOOST_TREE_TWEET_PAIR_TRAINING_DIR
from config import BOOST_TREE_TWEET_PAIR_TESTING_DIR


def main():
    print 'Starting program'
    train_hashtag_names = get_hashtag_file_names(SEMEVAL_HUMOR_TRAIN_DIR)
    param = {'max_depth': 50, 'eta': 1, 'silent': 0, 'objective': 'binary:logistic'}
    num_round = 2
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
    print np_data.shape
    print np_labels.shape

    dtrain = xgb.DMatrix(np_data, label=np_labels)
    bst = xgb.train(param, dtrain, num_round)
    np_preds = bst.predict(dtrain)
    np_preds_classified = np.round(np_preds)
    accuracy = np.mean(np_labels == np_preds_classified)
    print accuracy

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
    print 'Mean test accuracy: %s' % np.mean(list_of_accuracies)

if __name__ == '__main__':
    main()
