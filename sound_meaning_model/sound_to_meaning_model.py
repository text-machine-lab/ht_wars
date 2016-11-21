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
n_epochs = 25

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
    layer_size = 100

    tf_phonetic_emb = tf.placeholder(tf.float32, shape=[None, input_emb_size], name='phonetic_emb')
    tf_layer, _, _ = create_dense_layer(tf_phonetic_emb, input_emb_size, layer_size, activation='sigmoid')
    tf_layer, _, _ = create_dense_layer(tf_layer, layer_size, layer_size, activation='sigmoid')
    tf_layer, _, _ = create_dense_layer(tf_layer, layer_size, layer_size, activation='sigmoid')
    tf_layer, _, _ = create_dense_layer(tf_layer, layer_size, layer_size, activation='sigmoid')
    tf_layer, _, _ = create_dense_layer(tf_layer, layer_size, layer_size, activation='sigmoid')
    tf_layer, _, _ = create_dense_layer(tf_layer, layer_size, layer_size, activation='sigmoid')
    tf_layer, _, _ = create_dense_layer(tf_layer, layer_size, layer_size, activation='sigmoid')
    tf_layer, _, _ = create_dense_layer(tf_layer, layer_size, layer_size, activation='relu')
    tf_glove_emb_prediction, _, _ = create_dense_layer(tf_layer, layer_size, output_emb_size, activation=None)

    return [tf_phonetic_emb, tf_glove_emb_prediction]


def create_dense_layer(input_layer, input_size, output_size, activation=None):
    tf_w = tf.Variable(tf.random_normal([input_size, output_size], stddev=.1))
    tf_b = tf.Variable(tf.random_normal([output_size]))
    output_layer = tf.matmul(input_layer, tf_w) + tf_b
    if activation == 'relu':
        output_layer = tf.nn.relu(output_layer)
    elif activation == 'sigmoid':
        output_layer = tf.nn.sigmoid(output_layer)
    elif activation == None:
        pass
    else:
        print 'Did not specify layer activation or lack thereof'

    return output_layer, tf_w, tf_b


def build_sound_to_meaning_trainer(dataset, model_io):
    print 'Building trainer'
    glove_embs = dataset[1]
    glove_emb_size = glove_embs.shape[1]
    tf_glove_emb_prediction = model_io[1]
    tf_glove_emb_label = tf.placeholder(dtype=tf.float32, shape=[None, glove_emb_size], name='glove_emb_label')
    tf_batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')

    tf_loss = tf.reduce_mean(tf.squared_difference(tf_glove_emb_label, tf_glove_emb_prediction))

    return [tf_loss, tf_glove_emb_label, tf_batch_size]


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

    [glove_words, glove_embs, phonetic_embs] = dataset
    [tf_phonetic_emb, tf_glove_emb_predictions] = model_io
    [tf_loss, tf_glove_emb_label, tf_batch_size] = trainer_o

    m = phonetic_embs.shape[0]
    print m

    with tf.name_scope("SAVER"):
        saver = tf.train.Saver(max_to_keep=10)
    tf_train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.initialize_all_variables())
    for epoch in range(n_epochs):
        print 'Epoch %s' % epoch
        # Use these parameters to keep track of batch size and start location.
        starting_training_example = 0
        num_batches = m / batch_size
        remaining_batch_size = m % batch_size
        print 'Number of batches: %s' % num_batches
        batch_accuracies = []
        for i in range(num_batches+1):
            # If we are on the last batch, we are running leftover examples. Otherwise, we stick to global batch_size.
            if i == num_batches:
                current_batch_size = remaining_batch_size
            else:
                current_batch_size = batch_size

            glove_words_batch = glove_words[starting_training_example:starting_training_example + current_batch_size]
            phonetic_embs_batch = phonetic_embs[starting_training_example:starting_training_example + current_batch_size, :]
            glove_embs_batch = glove_embs[starting_training_example:starting_training_example + current_batch_size, :]
            # Run training step here.
            _, loss, glove_emb_predictions = sess.run([tf_train_op, tf_loss, tf_glove_emb_predictions],
                                                      feed_dict={tf_phonetic_emb: phonetic_embs_batch,
                                                                 tf_glove_emb_label: glove_embs_batch,
                                                                 tf_batch_size: current_batch_size})
            if i % 1000 == 0 and current_batch_size > 0:
                print 'Loss: %s' % loss
                mean_error = np.mean(np.abs(glove_embs_batch - glove_emb_predictions))
                print 'Mean error: %s' % mean_error
                mean_label_value = np.mean(np.abs(glove_embs_batch))
                max_label_value = np.max(np.abs(glove_embs_batch))
                mean_prediction_value = np.mean(np.abs(glove_emb_predictions))
                mean_phonetic_value = np.mean(np.abs(phonetic_embs_batch))
                print 'Mean glove embedding label value: %s' % mean_label_value
                print 'Max glove embedding label value: %s' % max_label_value
                print 'Mean glove embedding prediction value: %s' % mean_prediction_value
                print 'Mean input phonetic embedding value: %s' % mean_phonetic_value

            starting_training_example += current_batch_size
    return sess


def evaluate_model_performance(sess, dataset, model_io):
    print 'Evaluating model performance'

    print '100% boo ya'













if __name__ == '__main__':
    print 'Starting program'
    main()