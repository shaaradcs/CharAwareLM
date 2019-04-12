# CharAwareLM

Run pre_processing.py to create vocabularies from the training set and/or convert a plain text file to a file containing the data in a character/word index tensor representation.
Usage:
  python pre_processing.py [--build] [<file_name>]
The --build option forces building of vocabularies from the training set data/train.txt and dumping the vocabularies in the data folder.
If the option is not given, the vocabularies are built and dumped if they are not available in the data folder.
The <file_name> argument is used to specify which plain text file is to be converted to the index representation as a file '<file_name>.pkl'. If the argument is not given, a default value of 'data/train.txt' is used.

The code.py script reads train.txt.pkl and trains the language model on that data.

vocabulary.py contains the necessary methods to create vocabularies and is used by preprocessing.py.

While training, intermediate models are pickled in the model folder after each pass through the entire training set.
