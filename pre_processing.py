import pickle
import torch
import numpy as np
from vocabulary import *
import sys
import argparse

BUILD_DEFAULT='data/train.txt'
TARGET_DEFAULT='data/train.txt'
SEQ_LEN_DEFAULT=82
WORD_LENGTH=32

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seq_length', type=int, help='Optional sequence length', default=SEQ_LEN_DEFAULT)
parser.add_argument('-b', '--build', dest='build_file', nargs='?', const=BUILD_DEFAULT, default=None)
parser.add_argument('-t', '--target', dest='file_name', nargs='?', const=TARGET_DEFAULT, default=None)

args = parser.parse_args()

build_file = args.build_file
seq_length = args.seq_length
file_name = args.file_name

if build_file != None:
    to_build = True
else:
    to_build = False

if file_name != None:
    to_transform = True
else:
    to_transform = False

if to_build == True:
    # Vocabulary to be built
    char_vocabulary = CharVocabulary(build_file)
    word_vocabulary = WordVocabulary(build_file)
    print('Built vocabulary from file :', build_file)
    build = True
else:
    # Try to find already existing vocabulary
    to_build = False
    try:
        char_vocabulary = pickle.load(open('data/char_vocab.pkl', 'rb'))
        word_vocabulary = pickle.load(open('data/word_vocab.pkl', 'rb'))
        print('Loaded pre built vocabulary')
    except:
        # Unable to load vocabulary. To be built from default file
        to_build = True
        char_vocabulary = CharVocabulary(BUILD_DEFAULT)
        word_vocabulary = WordVocabulary(BUILD_DEFAULT)
        print('Built vocabulary from file :', BUILD_DEFAULT)


if to_transform:

    # Convert the given file into a suitable index representation described below:
    # File : List of sentences
    # Sentence : A tuple (tupl1, tupl1)
    # tupl1 : Tensor, each element of which corresonds to a character index representation of a word in the sentence
    # tupl2 : Tensor, each element of which is the index of a word in the sentence
    # If seq_length is non zero, each tensor is padded to sequence length with empty words

    to_break = True
    try:
        fp = open(file_name, 'r')
        print('Transforming file : ' + file_name)
    except:
        print('Unable to transform file : ' + file_name)
        to_break = False

    if not to_break:

        data = list()
        for line in fp.readlines():
            words = line.split()
            if seq_length == None or seq_length == 0:
                length = len(words)
            else:
                length = seq_length

            #### Character index tensor

            char_embed = torch.LongTensor(length + 2, WORD_LENGTH)
            for word_ind in range(-1, length + 1):
                if word_ind == -1:
                    word = (WORD_LENGTH - 2) * char_vocabulary.start_sentence_char
                elif word_ind == length:
                    word = ''
                elif word_ind < len(words):
                    word = words[word_ind]
                else:
                    word = ''
                char_embed_word = char_vocabulary.char_index_list(word)
                if len(char_embed_word) > WORD_LENGTH:
                    char_embed_word[31] = char_vocabulary.char_index(char_vocabulary.end_word_char)
                while len(char_embed_word) < WORD_LENGTH:
                    char_embed_word.append(char_vocabulary.char_index(char_vocabulary.padding_char))
                char_embed[word_ind + 1] = torch.from_numpy(np.array(char_embed_word[:WORD_LENGTH]))

            
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

# Dump the newly built vocabularies into files
if to_build == True:
    pickle.dump(char_vocabulary, open('data/char_vocab.pkl', 'wb'))
    pickle.dump(word_vocabulary, open('data/word_vocab.pkl', 'wb'))
    print('Saved built vocabulary')

