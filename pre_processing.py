import pickle
import torch
from vocabulary import *
import sys


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

    # Character index tensor
    char_embed = torch.LongTensor(len(words), 32)
    word_ind = 0
    for word in words:
        char_embed_word = char_vocabulary.char_index_list(word)
        while len(char_embed_word) < 32:
            char_embed_word.append(char_vocabulary.char_index(char_vocabulary.end_word_char))
        for i in range(0,len(char_embed_word)):
            char_embed[word_ind][i] = char_embed_word[i]
        word_ind += 1
    
    # Word index tensor
    word_embed = torch.LongTensor(len(words))
    for i in range(0, len(words)):
        word_embed[i] = word_vocabulary.word_index(words[i])

    # Append them a tuple to the data
    data.append( (char_embed, word_embed) )


# Dump the training file as a list of tensor tuples in a new file
pickle.dump(data, open(file_name + str('.pkl'), 'wb'))

# Dump the vocabularies into files as well, just in case they are needed
if build == True:
    pickle.dump(char_vocabulary, open('data/char_vocab.pkl', 'wb'))
    pickle.dump(word_vocabulary, open('data/word_vocab.pkl', 'wb'))

