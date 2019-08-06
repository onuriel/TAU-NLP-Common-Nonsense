TAU-NLP-Common-Nonsense

Project Description: 

Generating untrue sentences (non-sense) and creating a simple discriminator for commonsense and nonsense. 

Instructions for generating:

Download dataset first from https://github.com/commonsense/conceptnet5/wiki/Downloads - put this gz in the data folder, no need to extract it. Run data_preprocessing.py to generate a filtered cached version of the dataset. Download the checkpoint weights and put them in training_checkpoint folder. Now the model is ready to generate sentences!
run generate.py to generate sentences and put them in a file named generated_sentences.txt!

This work includes data from ConceptNet 5, which was compiled by the Commonsense Computing Initiative. ConceptNet 5 is freely available under the Creative Commons Attribution-ShareAlike license (CC BY SA 4.0) from http://conceptnet.io. The included data was created by contributors to Commonsense Computing projects, contributors to Wikimedia projects, Games with a Purpose, Princeton University's WordNet, DBPedia, OpenCyc, and Umbel.
