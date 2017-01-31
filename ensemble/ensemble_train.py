import os
import cPickle as pickle

import numpy as np

from sacred import Experiment

from tf_emb_char_humor.humor_model_evaluation import write_predictions_to_file
from tools import get_hashtag_file_names
from config import ENSEMBLE_DIR, SEMEVAL_HUMOR_EVAL_DIR, BOOST_TREE_TWEET_PAIR_EVAL_DIR, \
    BOOST_TREE_EVAL_TWEET_PAIR_PREDICTIONS, HUMOR_EVAL_PREDICTION_HASHTAGS, HUMOR_EVAL_PREDICTION_LABELS, \
    HUMOR_EVAL_TWEET_PAIR_PREDICTIONS, HUMOR_EVAL_PREDICTION_FIRST_TWEET_IDS, HUMOR_EVAL_PREDICTION_SECOND_TWEET_IDS, \
    ENSEMBLE_EVAL_PREDICTIONS_DIR
from config import HUMOR_TRAIN_PREDICTION_HASHTAGS, HUMOR_TRAIN_PREDICTION_LABELS, HUMOR_TRAIN_TWEET_PAIR_PREDICTIONS
from config import BOOST_TREE_TRAIN_TWEET_PAIR_PREDICTIONS
from config import SEMEVAL_HUMOR_TRIAL_DIR
from config import MONGO_ADDRESS

from feed_forward_network import FeedForwardNetwork
from boost_tree_humor.tree_model import XGBoostTreeModel

ex_name = 'ensemble'
ex = Experiment(ex_name)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data


def load_data_train():
    hashtag_names = load_pickle(HUMOR_TRAIN_PREDICTION_HASHTAGS)
    hashtag_labels = load_pickle(HUMOR_TRAIN_PREDICTION_LABELS)

    david_predictions = load_pickle(HUMOR_TRAIN_TWEET_PAIR_PREDICTIONS)
    boost_tree_predictions = load_pickle(BOOST_TREE_TRAIN_TWEET_PAIR_PREDICTIONS)

    assert len(hashtag_names) == len(hashtag_labels)
    assert len(hashtag_names) == len(david_predictions)
    assert len(david_predictions) == len(boost_tree_predictions)

    for i in range(len(hashtag_names)):
        assert len(david_predictions[i]) == len(boost_tree_predictions[i])

    predictions = []
    for i in range(len(hashtag_names)):
        pred_all = np.concatenate([david_predictions[i], np.reshape(boost_tree_predictions[i], (-1, 1)), ], axis=1)

        predictions.append(pred_all)

    return hashtag_names, hashtag_labels, predictions


def load_data_predict():
    hashtag_names = load_pickle(HUMOR_EVAL_PREDICTION_HASHTAGS)

    david_predictions = load_pickle(HUMOR_EVAL_TWEET_PAIR_PREDICTIONS)
    boost_tree_predictions = load_pickle(BOOST_TREE_EVAL_TWEET_PAIR_PREDICTIONS)

    first_tweet_ids = load_pickle(HUMOR_EVAL_PREDICTION_FIRST_TWEET_IDS)
    second_tweet_ids = load_pickle(HUMOR_EVAL_PREDICTION_SECOND_TWEET_IDS)

    assert len(hashtag_names) == len(david_predictions)
    assert len(david_predictions) == len(boost_tree_predictions)

    for i in range(len(hashtag_names)):
        assert len(david_predictions[i]) == len(boost_tree_predictions[i])

    predictions = []
    for i in range(len(hashtag_names)):
        pred_all = np.concatenate([david_predictions[i], np.reshape(boost_tree_predictions[i], (-1, 1)), ], axis=1)

        predictions.append(pred_all)

    return hashtag_names, predictions, first_tweet_ids, second_tweet_ids


xgboost_params = {
    'objective': 'binary:logistic',
    'max_depth': 8,
    'eta': 0.038,  # learning_rate
    'gamma': 2,  # min_split_loss
    'reg_lambda': 1.03e-05,
    'num_round': 20,
    'silent': 1,
}


@ex.config
def config():
    nb_epoch = 50
    batch_size = 512
    verbose = 1

    layer_size = 200
    num_layers = 5
    regularization = 0.00005

    # checkpoint_filename = 'ensemble_weights.hdf5'
    checkpoint_filename = 'ensemble_weights_xgboost.bin'


@ex.main
def main(layer_size, num_layers, regularization, checkpoint_filename, nb_epoch, batch_size, verbose):
    hashtag_names, hashtag_labels, y_pred_train = load_data_train()
    print 'Hashtags:', len(hashtag_names)

    # validation_hashtags = ['Bad_Inventions', 'Fast_Food_Books', 'Gentler_Songs', 'My_Family_In_4_Words', 'Spooky_Bands', ]
    # validation_hashtags = ['Make_A_Movie_Sick', 'Florida_A_Movie', 'Drunk_Books', 'Ruin_Shakespeare', 'Comic_Book_TV_Shows']
    # validation_hashtags = np.random.permutation(hashtag_names)[:5]
    validation_hashtags = []

    data_train = [y_pred_train[i] for i, h in enumerate(hashtag_names) if h not in validation_hashtags]
    labels_train = [hashtag_labels[i] for i, h in enumerate(hashtag_names) if h not in validation_hashtags]

    data_val = [y_pred_train[i] for i, h in enumerate(hashtag_names) if h in validation_hashtags]
    labels_val = [hashtag_labels[i] for i, h in enumerate(hashtag_names) if h in validation_hashtags]

    data_train = np.concatenate(data_train, axis=0)
    labels_train = np.concatenate(labels_train, axis=0)

    if len(validation_hashtags) > 0:
        data_val = np.concatenate(data_val, axis=0)
        labels_val = np.concatenate(labels_val, axis=0)
    else:
        data_val = None
        labels_val = None

    # data_train = data_train[:10000]
    # labels_train = labels_train[:10000]

    print 'Data:', data_train.shape, labels_train.shape
    if data_val is not None and labels_val is not None:
        print 'Data val:', data_val.shape, labels_val.shape

    # train the model
    model = XGBoostTreeModel(**xgboost_params)

    model.train(data_train, labels_train)

    model.save_model(os.path.join(ENSEMBLE_DIR, checkpoint_filename))

    y_pred_train = model.predict(data_train)
    y_pred_train_classified = np.round(y_pred_train)
    accuracy_train = np.mean(labels_train == y_pred_train_classified)

    if data_val is not None and labels_val is not None:
        y_pred_val = model.predict(data_val)
        y_pred_val_classified = np.round(y_pred_val)
        accuracy_val = np.mean(labels_val == y_pred_val_classified)
    else:
        accuracy_val = -1

    # params = {
    #     'input_dim': data_train.shape[1],
    #     'layer_size': layer_size,
    #     'num_layers': num_layers,
    #     'regularization': regularization,
    #     'checkpoint_filename': os.path.join(ENSEMBLE_DIR, checkpoint_filename),
    # }
    #
    # model = FeedForwardNetwork(**params)
    # model.fit(data_train, labels_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose)
    #
    # # evaluate on the training and val data
    # y_pred_train = model.predict(data_train, batch_size=batch_size, verbose=verbose)
    # accuracy_train = np.mean(labels_train == y_pred_train)
    #
    # y_pred_val = model.predict(data_val, batch_size=batch_size, verbose=verbose)
    # accuracy_val = np.mean(labels_val == y_pred_val)

    print 'Accuracy train:', accuracy_train, 'Accuracy Val:', accuracy_val

    result = {
        'accuracy_train': accuracy_train,
        'accuracy_val': accuracy_val
    }

    return result


@ex.command
def predict(layer_size, num_layers, regularization, checkpoint_filename, nb_epoch, batch_size, verbose):
    if not os.path.isdir(ENSEMBLE_EVAL_PREDICTIONS_DIR):
        os.makedirs(ENSEMBLE_EVAL_PREDICTIONS_DIR)

    hashtag_names, data_all, first_tweet_ids, second_tweet_ids = load_data_predict()

    model = XGBoostTreeModel(**xgboost_params)
    model.restore_model(os.path.join(ENSEMBLE_DIR, checkpoint_filename))

    # predict on each hashtag

    hashtag_predictions = []
    for i, hashtag_name in enumerate(hashtag_names):
        np_data = data_all[i]
        first_tweet = first_tweet_ids[i]
        second_tweet = second_tweet_ids[i]

        predictions = model.predict(np_data)
        predictions_classified = np.round(predictions).astype(np.int32)

        perdictions_filename = os.path.join(ENSEMBLE_EVAL_PREDICTIONS_DIR, hashtag_name + '_PREDICT.tsv')
        write_predictions_to_file(perdictions_filename, predictions_classified, first_tweet, second_tweet)

        print 'Hashtag', i, hashtag_name, 'Predictions saved:', perdictions_filename


        # params = {
        #     'input_dim': np_data.shape[1],
        #     'layer_size': layer_size,
        #     'num_layers': num_layers,
        #     'regularization': regularization,
        #     'checkpoint_filename': os.path.join(ENSEMBLE_DIR, checkpoint_filename),
        # }
        #
        # model = FeedForwardNetwork(**params)
        #
        # model.restore()
        #
        # y_pred_train = model.predict(np_data, batch_size=batch_size, verbose=verbose)
        # accuracy_train = np.mean(np_labels == y_pred_train)
        #
        # print 'Accuracy restored:', accuracy_train


if __name__ == '__main__':
    ex.run_commandline()
