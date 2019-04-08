# CharAwareLM

Run pre_processing.py to create vocabularies from the training set.
The vocabulary is then used to transform ./data/train.txt into its character and word indices and dumped as ./data/train.p

The code.py script reads train.p and trains the language model on that data.

vocabulary.py contains the necessary methods to create vocabularies and is used by preprocessing.py.

While training, intermediate models are pickled in the model folder after each pass through the entire training set.
