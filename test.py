import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import time
from tqdm import tqdm
import module
import sys

# Import model from file
if len(sys.argv) > 1:
    model_file = sys.argv[1]
else:
    model_file = 'model/epoch_12.pkl'
model = pickle.load(open(model_file, 'rb'))
print('Using model from file : ', model_file)

# Obtain data on which to test the model
if len(sys.argv) > 2:
    data_file = sys.argv[2]
else:
    data_file = 'data/test.txt.pkl'
data = pickle.load(open(data_file, 'rb'))
print('Using data from file : ', data_file)

if torch.cuda.is_available():
    model.cuda()
    model = nn.DataParallel(model)
# Loss
criterion = nn.CrossEntropyLoss()

test_loss = torch.zeros(1).float().cuda()
# Calculate the perplexity on the validation set
with torch.no_grad():
    perplexity = torch.zeros(1).float().cuda()
    for line in tqdm(data):
        if torch.cuda.is_available():
            char_index = line[0].view(1, line[0].shape[0], -1).cuda()
            word_index = line[1].cuda()
        else:
            char_index = line[0].view(1, line[0].shape[0], -1)
            word_index = line[1]
        result = model(char_index)
        loss = criterion(result[0], word_index[1:word_index.shape[0]-1])
        test_loss += loss
        perplexity += torch.exp(loss)
    test_loss = test_loss / len(data)
    perplexity = perplexity / len(data)

    # Print statistics
    print('Loss : ' + str(test_loss))
    print('Perplexity : ' + str(perplexity))
    print('Number of lines : ' + str(len(data)))
