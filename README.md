# TAU-NLP-Common-Nonsense

## Project Description: 
Generating untrue sentences (non-sense) and creating a simple discriminator for commonsense and nonsense. 

## Instructions:
### Preprocess the Data:
1. Download dataset first from [here](https://github.com/commonsense/conceptnet5/wiki/Downloads) and put the gz file in the *data* folder, no need to extract it.
2. Run *data_preprocessing.py* to generate a filtered cached version of the dataset and a sequence-to-sentence dataset in the *out* folder.

### Train the Model:
1. Just run *train.py*.
2. Alternatively, you can download a trained model (checkpoints) and place it in the *training_checkpoints* folder.

### Generate Sentences:
1. Just run *generator.py*. The output will be available in the *out* folder (unless configured differently).

> This work includes data from ConceptNet 5, which was compiled by the Commonsense Computing Initiative. ConceptNet 5 is freely available under the Creative Commons Attribution-ShareAlike license (CC BY SA 4.0) from http://conceptnet.io. The included data was created by contributors to Commonsense Computing projects, contributors to Wikimedia projects, Games with a Purpose, Princeton University's WordNet, DBPedia, OpenCyc, and Umbel.
