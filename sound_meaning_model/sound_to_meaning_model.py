'''David Donahue 2016. This script runs the sound-to-meaning model. The model takes in as input
a phonetic embedding that can be used to pronounce a word. It outputs a prediction of the GloVe
vector for that word. So in a sense, the model can hear what a word sounds like it means. The resultant
embedding will be used as a feature to humor model. This script imports the dataset for the model,
builds the model, trains the model, and evaluates the model's performance.'''

import tensorflow as tf
import numpy as np
import cPickle as pickle
import os

os.environ['GLOG_minloglevel'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

model_path = 'saved_models/'
batch_size = 100
regularization_constant = 0.1
n_epochs = 500


def main():
    dataset = load_phonetic_and_glove_embeddings_dataset()
    model_io = build_sound_to_meaning_model(dataset)
    trainer_io = build_sound_to_meaning_trainer(dataset, model_io)
    sess = train_sound_to_meaning_model(dataset, model_io, trainer_io)
    evaluate_model_performance(sess, dataset, model_io)


def load_phonetic_and_glove_embeddings_dataset(directory=''):
    print 'Loading dataset'
    glove_words = pickle.load(open(directory + 'glove_words.cpkl', 'rb'))
    glove_embs = np.load(directory + 'glove_embs.npy')
    phonetic_embs = np.load(directory + 'phonetic_embs.npy')
    print phonetic_embs.shape

    return [glove_words, glove_embs, phonetic_embs]


def build_sound_to_meaning_model(dataset):
    print 'Building sound-to-meaning model'
    '''Builds a model to convert a phonetic embedding for a word
    into an approximation of its glove embedding. Uses four
    fully-connected layers that decrease in output size from 400
    to 200 in 50 neuron increments. Function returns a phonetic embeddings
    tensor as the model input and a glove embedding prediction tensor as
    the model output.'''
    glove_embs = dataset[1]
    phonetic_embs = dataset[2]
    input_emb_size = phonetic_embs.shape[1]
    output_emb_size = glove_embs.shape[1]
    layer_size = 400

    tf_phonetic_emb = tf.placeholder(tf.float32, shape=[None, input_emb_size], name='phonetic_emb')
    tf_layer, _, _ = create_dense_layer(tf_phonetic_emb, input_emb_size, layer_size, activation='relu')

    num_hidden_layers = 15
    for i in range(num_hidden_layers):
        tf_layer, _, _ = create_dense_layer(tf_layer, layer_size, layer_size, activation='relu')

    tf_glove_emb_prediction, _, _ = create_dense_layer(tf_layer, layer_size, output_emb_size, activation=None)

    tf_batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')
    tf_constant_guess_bias = tf.Variable(tf.random_normal([output_emb_size]))
    tf_guess_emb = tf.zeros([tf_batch_size, output_emb_size])
    tf_guess_emb = tf_guess_emb + tf_constant_guess_bias

    return [tf_phonetic_emb, tf_glove_emb_prediction, tf_batch_size, tf_constant_guess_bias]  # tf_glove_emb_prediction, tf_batch_size]


def create_dense_layer(input_layer, input_size, output_size, activation=None):
    tf_w = tf.Variable(tf.random_normal([input_size, output_size], stddev=.1))
    tf_b = tf.Variable(tf.random_normal([output_size]))
    output_layer = tf.matmul(input_layer, tf_w) + tf_b
    if activation == 'relu':
        output_layer = tf.nn.relu(output_layer)
    elif activation == 'sigmoid':
        output_layer = tf.nn.sigmoid(output_layer)
    elif activation is None:
        pass
    else:
        print 'Did not specify layer activation'

    return output_layer, tf_w, tf_b


def build_sound_to_meaning_trainer(dataset, model_io):
    print 'Building trainer'
    glove_embs = dataset[1]
    glove_emb_size = glove_embs.shape[1]
    tf_glove_emb_prediction = model_io[1]
    tf_batch_size = model_io[2]
    tf_glove_emb_label = tf.placeholder(dtype=tf.float32, shape=[None, glove_emb_size], name='glove_emb_label')

    tf_loss_per_example = tf.reduce_mean(tf.squared_difference(tf_glove_emb_label, tf_glove_emb_prediction), reduction_indices=[1])

    tf_index_of_highest_loss = tf.argmax(tf_loss_per_example, 0)

    tf_loss = tf.reduce_mean(tf_loss_per_example)

    return [tf_loss, tf_glove_emb_label, tf_index_of_highest_loss]


def regularization_cost(reg_constant):
    vars = tf.trainable_variables()
    loss = 0
    for var in vars:
        loss += tf.nn.l2_loss(var)
    return loss * reg_constant


def train_sound_to_meaning_model(dataset, model_io, trainer_o, learning_rate=.001):
    print 'Training model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    ######################################
    print("Creating Tensorboard visualization.")
    writer = tf.train.SummaryWriter("/tmp/s2m/")
    writer.add_graph(tf.get_default_graph())
    ######################################

    [glove_words, glove_embs, phonetic_embs] = dataset
    [tf_phonetic_emb, tf_glove_emb_predictions, tf_batch_size, tf_constant_guess_bias] = model_io
    [tf_loss, tf_glove_emb_label, tf_index_of_highest_loss] = trainer_o

    m = phonetic_embs.shape[0]

    with tf.name_scope("SAVER"):
        saver = tf.train.Saver(max_to_keep=10)
    tf_train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.initialize_all_variables())
    for epoch in range(n_epochs):
        print 'Epoch %s' % epoch

        # Shuffle dataset.
        random_state = np.random.get_state()
        np.random.shuffle(glove_words)
        np.random.set_state(random_state)
        np.random.shuffle(glove_embs)
        np.random.set_state(random_state)
        np.random.shuffle(phonetic_embs)

        # Use these parameters to keep track of batch size and start location.
        starting_training_example = 0
        num_batches = m / batch_size
        remaining_batch_size = m % batch_size
        batch_losses = []
        batch_mean_errors = []
        print 'Number of batches: %s' % num_batches
        for i in range(num_batches+1):
            # If we are on the last batch, we are running leftover examples. Otherwise, we stick to global batch_size.
            if i == num_batches:
                current_batch_size = remaining_batch_size
            else:
                current_batch_size = batch_size

            if current_batch_size > 0:
                glove_words_batch = glove_words[starting_training_example:starting_training_example + current_batch_size]
                phonetic_embs_batch = phonetic_embs[starting_training_example:starting_training_example + current_batch_size, :]
                glove_embs_batch = glove_embs[starting_training_example:starting_training_example + current_batch_size, :]
                # Run training step here.
                _, loss, glove_emb_predictions, index_highest_loss, constant_guess_bias = sess.run([tf_train_op,
                                                                                                    tf_loss,
                                                                                                    tf_glove_emb_predictions,
                                                                                                    tf_index_of_highest_loss,
                                                                                                    tf_constant_guess_bias],
                                                                                                    feed_dict={tf_phonetic_emb: phonetic_embs_batch,
                                                                                                               tf_glove_emb_label: glove_embs_batch,
                                                                                                               tf_batch_size: current_batch_size})
                # print index_highest_loss.shape
                # print 'Highest loss: %s' % glove_words_batch[index_highest_loss]
                # if i % 1000 == 0:
                #     print 'Constant guess, average embedding:'
                #     print constant_guess_bias
                #     average_emb = np.mean(glove_embs_batch, axis=0)
                #     print average_emb
                #     print constant_guess_bias - average_emb
                mean_error = np.mean(np.abs(glove_embs_batch - glove_emb_predictions))
                batch_losses.append(loss)
                batch_mean_errors.append(mean_error)
                # if i % 500 == 0:
                #     print 'Mean error: %s' % mean_error
                #     mean_label_value = np.mean(np.abs(glove_embs_batch))
                #     max_label_value = np.max(np.abs(glove_embs_batch))
                #     mean_prediction_value = np.mean(np.abs(glove_emb_predictions))
                #     mean_phonetic_value = np.mean(np.abs(phonetic_embs_batch))
                #     mean_phonetic_std = np.mean(np.std(phonetic_embs_batch))
                #     print 'Mean glove embedding label value: %s' % mean_label_value
                #     print 'Max glove embedding label value: %s' % max_label_value
                #     print 'Mean glove embedding prediction value: %s' % mean_prediction_value
                #     print 'Mean input phonetic embedding value: %s' % mean_phonetic_value
                #     print 'Mean input phonetic embedding std: %s' % mean_phonetic_std

                starting_training_example += current_batch_size
        epoch_loss = np.mean(batch_losses)
        epoch_mean_error = np.mean(batch_mean_errors)
        print 'Epoch loss: %s' % epoch_loss
        print 'Epoch mean error: %s' % epoch_mean_error

    return sess


def find_index_of_largest_number(list_num):
    '''This function finds the largest number in the list
    list_num, and returns its index. This function
    is intended to print out the examples the model is doing worst on.'''
    highest_value = list_num[0]
    index_of_highest_value = 0
    for index in range(len(list_num)):
        number = list_num[index]
        if number > highest_value:
            highest_value = number
            index_of_highest_value = index
    return index


def evaluate_model_performance(sess, dataset, model_io):
    print 'Evaluating model performance'

    print '100% boo ya'













if __name__ == '__main__':
    print 'Starting program'
    main()