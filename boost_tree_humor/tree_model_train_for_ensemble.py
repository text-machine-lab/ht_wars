import cPickle as pickle

import numpy as np

from tf_emb_char_humor.humor_ensemble_processing import num_groups
from boost_tree_humor.tree_model import XGBoostTreeModel
from config import HUMOR_TRAIN_PREDICTION_HASHTAGS, BOOST_TREE_TWEET_PAIR_TRAIN_DIR, \
    BOOST_TREE_TRAIN_TWEET_PAIR_PREDICTIONS

xgboost_params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'eta': 0.038,  # learning_rate
    'gamma': 2,  # min_split_loss
    'reg_lambda': 1.03e-05,
    'num_round': 10,
    'silent': 1,
}


def get_hashtag_names():
    with open(HUMOR_TRAIN_PREDICTION_HASHTAGS, 'rb') as f:
        hashtag_names = pickle.load(f)

    return hashtag_names


def get_train_data(hashtag_names, current_group_hashtags):
    train_hashtags = [h for h in hashtag_names if h not in current_group_hashtags]

    data_train, labels_train = load_tree_data(train_hashtags)

    return data_train, labels_train


def load_tree_data(hashtags):
    if not isinstance(hashtags, list):
        hashtags = [hashtags]

    list_of_labels = []
    list_of_datas = []
    for hashtag_name in hashtags:
        np_hashtag_data = np.load(open(BOOST_TREE_TWEET_PAIR_TRAIN_DIR + hashtag_name + '_data.npy', 'rb'))
        list_of_datas.append(np_hashtag_data)

        np_hashtag_labels = np.load(open(BOOST_TREE_TWEET_PAIR_TRAIN_DIR + hashtag_name + '_labels.npy', 'rb'))
        list_of_labels.append(np_hashtag_labels)

    data_train = np.vstack(list_of_datas)
    labels_train = np.concatenate(list_of_labels, axis=0)

    return data_train, labels_train


def main():
    hashtag_names = get_hashtag_names()
    print 'Hashtag names:', len(hashtag_names), 'num groups:', num_groups

    hashtag_predictions = []
    for hashtag_group_index in range(num_groups):
        num_hashtags = len(hashtag_names)
        num_hashtags_in_group = num_hashtags / num_groups + 1
        starting_hashtag_index = num_hashtags_in_group * hashtag_group_index
        hashtags_in_group = hashtag_names[starting_hashtag_index:starting_hashtag_index + num_hashtags_in_group]

        print 'Group:', hashtag_group_index, 'hashtags:', len(hashtags_in_group)

        # load the data
        data_train, labels_train = get_train_data(hashtag_names,
                                                                                hashtags_in_group)
        print 'Group:', hashtag_group_index, 'data:', data_train.shape, labels_train.shape

        # train the model
        model = XGBoostTreeModel(**xgboost_params)

        model.train(data_train, labels_train)

        accuracy_train = model.evaluate(data_train, labels_train)
        print 'Group:', hashtag_group_index, 'accuracy train:', accuracy_train


        # get the predictions on individual hashtags
        hashtags_in_group_accuracies = []
        for hashtag_name in hashtags_in_group:
            data_predict, labels_predict = load_tree_data(hashtag_name)

            predictions = model.predict(data_predict)
            predictions_classified = np.round(predictions)
            accuracy_predict = np.mean(labels_predict == predictions_classified)

            hashtags_in_group_accuracies.append(accuracy_predict)

            hashtag_predictions.append(predictions_classified)

        print 'Group:', hashtag_group_index, 'mean accuracy:', np.mean(hashtags_in_group_accuracies)

    # save the predictions from the boost tree model
    with open(BOOST_TREE_TRAIN_TWEET_PAIR_PREDICTIONS, 'wb') as f:
        pickle.dump(hashtag_predictions, f)

    print 'Predictions saved:', BOOST_TREE_TRAIN_TWEET_PAIR_PREDICTIONS

if __name__ == '__main__':
    main()
