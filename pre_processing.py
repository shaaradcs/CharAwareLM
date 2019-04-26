import pickle
import torch
import numpy as np
from vocabulary import *
import sys

seq_length = 82

if len(sys.argv) == 2 and sys.argv[1] == '--build':
    # Vocabulary to be built
    char_vocabulary = CharVocabulary('data/train.txt')
    word_vocabulary = WordVocabulary('data/train.txt')
    build = True
else:
    # Try to find already existing vocabulary
    build = False
    try:
        char_vocabulary = pickle.load(open('data/char_vocab.pkl', 'rb'))
        word_vocabulary = pickle.load(open('data/word_vocab.pkl', 'rb'))
    except:
        # No vocabulary exists. To be built
        build = True
        char_vocabulary = CharVocabulary('data/train.txt')
        word_vocabulary = WordVocabulary('data/train.txt')


################################################################################
# Convert the training file into a suitable index representation described below:
# File : List of sentences
# Sentence : A tuple (tupl1, tupl1)
# tupl1 : Tensor, each element of which corresonds to a character index representation of a word in the sentence
# tupl2 : Tensor, each element of which is the index of a word in the sentence

# File to be transformed into a suitable representation
if len(sys.argv) == 3:
    file_name = sys.argv[2]
elif len(sys.argv) == 2 and sys.argv[1] != '--build':
        file_name = sys.argv[1]
else:
    file_name = 'data/train.txt'

try:
    fp = open(file_name)
    print('Transforming file : ' + file_name)
except:
    file_name = 'data/train.txt'
    fp = open(file_name)
    print('Transforming file : ' + file_name)


data = list()
for line in fp.readlines():
    words = line.split()
    if seq_length == None:
        length = len(words)
    else:
        length = seq_length

    #### Character index tensor

    char_embed = torch.LongTensor(length + 2, 32)
    for word_ind in range(-1, length + 1):
        if word_ind == -1:
            word = 30 * char_vocabulary.start_sentence_char
        elif word_ind == length:
            word = ''
        elif word_ind < len(words):
            word = words[word_ind]
        else:
            word = ''
        char_embed_word = char_vocabulary.char_index_list(word)
        if len(char_embed_word) > 32:
            char_embed_word[31] = char_vocabulary.char_index(char_vocabulary.end_word_char)
        while len(char_embed_word) < 32:
            char_embed_word.append(char_vocabulary.char_index(char_vocabulary.padding_char))
        char_embed[word_ind + 1] = torch.from_numpy(np.array(char_embed_word[:32]))

    
    # Word index tensor
    word_embed = torch.LongTensor(length + 2)
    for i in range(-1, length + 1):
        if i == -1:
            word = ''
        elif i == length:
            word = ''
        elif i < len(words):
            word = words[i]
        else:
            word = ''
        word_embed[i + 1] = word_vocabulary.word_index(word)

    # Append them a tuple to the data
    data.append( (char_embed, word_embed) )



# Dump the input file as a list of tensor tuples in a new file
pickle.dump(data, open(file_name + str('.pkl'), 'wb'))

# Dump the vocabularies into files as well, just in case they are needed
if build == True:
    pickle.dump(char_vocabulary, open('data/char_vocab.pkl', 'wb'))
    pickle.dump(word_vocabulary, open('data/word_vocab.pkl', 'wb'))

