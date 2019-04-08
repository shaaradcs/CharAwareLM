import pickle
import torch
from vocabulary import *

char_vocabulary = CharVocabulary('data/train.txt')
word_vocabulary = WordVocabulary('data/train.txt')

################################################################################
# Convert the training file into a suitable index representation described below:
# File : List of sentences
# Sentence : A tuple (tupl1, tupl1)
# tupl1 : Tensor, each element of which corresonds to a character index representation of a word in the sentence
# tupl2 : Tensor, each element of which is the index of a word in the sentence

fp = open('data/train.txt')
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
pickle.dump(data, open('data/train.p', 'wb'))

# Dump the vocabularies into files as well, just in case they are needed
pickle.dump(char_vocabulary, open('data/char_vocab.p', 'wb'))
pickle.dump(word_vocabulary, open('data/word_vocab.p', 'wb'))

