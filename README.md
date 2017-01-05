# ht_wars
Repository for Hashtag Wars Project

Code for supervised and unsupervised systems.

THIS REPOSITORY CONTAINS CODE FOR VARIOUS MODELS AND SCRIPTS RELATED TO THE #HashTagWars DATASET.

- The first CNN and LSTM character model scripts to make predictions on the dataset can be found in /char_model/

- Input/output data for all models can be found in /data/

- Data paths to files/directories in the directory /data can be imported from the config.py module

- Script for a model that can convert words/characters to phonemes/phonetic embeddings can be found in /phoneme_model/

- Script for an updated #HashtagWars humor model that takes phonetic/GloVe embedding features can be found in /humor_model/

- An in-development model to convert phonetic embeddings to Glove embedding approximations can be found in /sound_meaning_model/ (not working)


DATA DOWNLOAD AND CREATION REQUIREMENTS:

- The file 'glove.twitter.27B' must be downloaded (http://nlp.stanford.edu/projects/glove/) and copied into the /data/ folder

- The directories 'train_dir' and 'trial dir' must be downloaded (http://alt.qcri.org/semeval2017/task6/index.php?id=data-and-tools)
    and copied into the /data/ folder

- Run 'python char_model/ht_wars_data_processing.py' to create tweet pair data for the character-based humor models, found in /data/numpy_tweet_pairs/

- The files cmudict-0.7b.symbols.txt and cmudict-0.7b.txt must be downloaded (http://www.speech.cs.cmu.edu/cgi-bin/cmudict) and copied into the /data/ folder

- Run 'python phoneme_model/char2phone_processing.py extract_dataset' to create dictionaries (cmu_char_to_index.cpkl, cmu_phone_to_index.cpkl)
    and data needed to train the phoneme model (cmu_words.npy, cmu_pronunciations.npy)

- Run 'python phoneme_model/char2phone_model.py' to train phoneme models. These models are saved in /data/char_2_phone_models/

Once data is generated, all functions in tools.py and tf_tools.py should work. Relative paths from subfolders to datafiles can be found in config.py module.
If names of external data are to change (possibly due to a new version, etc.), the paths in config.py can be changed locally before data creation begins.

ATTENTION: All scripts in subfolders of ht_wars access the config.py script for data paths. This functionality is available automatically in some IDE's (ie. Pycharm),
but is not enabled by default if running script from command line. The ht_wars directory must be added to the python path!!!

This README last updated: 26 December 2016