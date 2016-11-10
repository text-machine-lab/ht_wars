'''David Donahue 2016. This script runs the sound-to-meaning model. The model takes in as input
a phonetic embedding that can be used to pronounce a word. It outputs a prediction of the GloVe
vector for that word. So in a sense, the model can hear what a word sounds like it means. The resultant
embedding will be used as a feature to humor model. This script imports the dataset for the model,
builds the model, trains the model, and evaluates the model's performance.'''

import tensorflow as tf
import numpy as np


def main():
    dataset = load_phonetic_and_glove_embeddings_dataset()
    model_io = build_sound_to_meaning_model(dataset)
    trainer_o = build_sound_to_meaning_trainer(dataset, model_io)
    evaluate_model_performance(dataset, model_io, trainer_o)


def load_phonetic_and_glove_embeddings_dataset():
    print 'Loading dataset'

    return ['word', 'phoneme_emb', 'glove_emb']


def build_sound_to_meaning_model(dataset):
    print 'Building sound-to-meaning model'
    return ['model_inputs', 'model_outputs']


def build_sound_to_meaning_trainer(dataset, model_io):
    print 'Building trainer'
    return ['training_step']


def evaluate_model_performance(dataset, model_io, trainer_o):
    print 'Evaluating model performance'
    print '100% boo ya'













if __name__ == '__main__':
    print 'Starting program'
    main()