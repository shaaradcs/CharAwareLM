import pickle

###################################################
#   Preprocessing characters and words to create vocabularies from the training set
#
#   Assumptions:
#       All lower case alphabets occur at least once in the training set
#   Comments:
#       All characters not in the training set are treated as unknown characters


class CharVocabulary:
    def __init__(self, file_name):
        fp = open(file_name)
        text = fp.read()

        # Obtain list of all characters in the training file
        char_list = list()
        for char in text:
            if char not in char_list:
                char_list.append(char)
        char_list.sort()

        #### Add 'start-of-word', 'end-of-word' and 'unknown' characters
        # Start-of-word
        for i in range(0, 255):
            # Check if ascii(i) is a character already used in the vocabulary
            if chr(i) not in char_list:
                char_list.insert(0, chr(i))
                break
        # End-of-word
        for i in range(0, 255):
            # Check if ascii(i) is a character already used in the vocabulary
            if chr(i) not in char_list:
                char_list.append(chr(i))
                break
        # Unknown character
        for i in range(0, 255):
            # Check if ascii(i) is a character already used in the vocabulary
            if chr(i) not in char_list:
                char_list.append(chr(i))
                break

        ###### Convert the list into a dictionary for easy access

        # char_vocab : Dictionary which maps each character in the vocabulary to an integer
        # The integer acts as an index corresponding to the character
        self.char_vocab = dict()
        count = 0
        for char in char_list:
            self.char_vocab[char] = count
            count += 1
        self.start_word_char = char_list[0]
        self.end_word_char = char_list[len(char_list) - 2]
        self.unknown_char = char_list[len(char_list) - 1]

    # Function to convert a character to its index as defined by char_vocab
    def char_index(self, char):
        if char in self.char_vocab.keys():
            return self.char_vocab[char]
        if ('A' <= char) and (char <= 'Z'):
            # Convert to a lower case character and return its index
            char += (ord('a') - ord('A'))
            return self.char_vocab[char]
        # Return as unknown character
        return self.char_vocab[self.unknown_char]

    # Function to convert a word into a list of character indices
    def char_index_list(self, word):
        lst = list()
        # Start-of-word character
        lst.append(self.char_vocab[self.start_word_char])
        for char in word:
            lst.append(self.char_index(char))
        # End-of-word character
        lst.append(self.char_vocab[self.end_word_char])
        return lst


class WordVocabulary:
    def __init__(self, file_name):
        fp = open(file_name)
        text = fp.read()

        ################# Create dictionary of words ##########

        word_seq = text.split()
        word_list = list()
        for word in word_seq:
            if word not in word_list:
                word_list.append(word)

        # Create word vocabulary as a dictionary
        self.word_vocab = dict()
        count = 0
        for word in word_list:
            self.word_vocab[word] = count
            count += 1

    # Function to convert a word to its index
    def word_index(self, word):
        if word in self.word_vocab:
            return self.word_vocab[word]
        return len(self.word_vocab.keys())


