import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import time
from tqdm import tqdm
import module

"""
if torch.cuda.is_available():
    print('GPU available')
    print('Current(default) GPU device : ' + str(torch.cuda.current_device()))
    device = 1
    torch.cuda.set_device(device)
    print('GPU device set to ' + str(torch.cuda.current_device()))
else:
    print('GPU unavailable. Using CPU')
"""

# Initialize model
model = module.LanguageModel()
model.init_weights()

if torch.cuda.is_available():
    model.cuda()
    model = nn.DataParallel(model)


# Loading training data in the form of list of tensor tuples
data = pickle.load(open('data/train.txt.pkl', 'rb'))
print('Training data loaded')

# valid_data = pickle.load(open('data/valid.txt.pkl', 'rb'))
valid_data = pickle.load(open('data/valid.txt.pkl', 'rb'))
print('Validation data loaded')

learning_rate = 1
batch_size = 20

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Load model
# model = pickle.load(open('model/epoch_11.pkl', 'rb'))

for epoch in range(0, 30):

    # Pass through the entire training data once, in batches
    # Also, calculate the average loss over all sentences
    print('Training')
    training_loss = torch.zeros(1).float().cuda()
    for step in tqdm(range(0, len(data)//batch_size)):
        offset = (step * batch_size) % (len(data) - batch_size)
        batch_data = data[offset:(offset + batch_size)]
        if torch.cuda.is_available():
            char_index = torch.stack([batch_data[i][0] for i in range(len(batch_data))]).cuda()
            word_index = torch.stack([batch_data[i][1] for i in range(len(batch_data))]).cuda()
            train_loss = torch.zeros(1, requires_grad=True).cuda()
        else:
            char_index = torch.stack([batch_data[i][0] for i in range(len(batch_data))])
            word_index = torch.stack([batch_data[i][1] for i in range(len(batch_data))])
            train_loss = torch.zeros(1, requires_grad=True)
        optim.zero_grad()
        results_1, results_2 = model(char_index)
        for i in range(0, batch_size):
            train_loss += criterion(results_1[i][1:word_index[i].shape[0]-1], word_index[i][2:word_index[i].shape[0]])
            train_loss += criterion(results_2[i][1:word_index[i].shape[0]-1], word_index[i][0:word_index[i].shape[0]-2])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        training_loss += train_loss
    training_loss = training_loss / (len(data) // batch_size)
    
    print('Validation')
    validation_loss = torch.zeros(1).float().cuda()
    # Calculate the perplexity on the validation set
    with torch.no_grad():
        if torch.cuda.is_available():
            perplexity = torch.zeros(1).float().cuda()
        else:
            perplexity = torch.zeros(1).float()
        for line in tqdm(valid_data):
            char_index = line[0].view(1, line[0].shape[0], -1).cuda()
            word_index = line[1].cuda()
            result_1, result_2 = model(char_index)
            loss = criterion(result_1[0][1:word_index.shape[0]-1], word_index[2:word_index.shape[0]])
            loss += criterion(result_2[0][1:word_index.shape[0]-1], word_index[0:word_index.shape[0]-2])
            validation_loss += loss
            perplexity += torch.exp(loss)
        validation_loss = validation_loss / len(valid_data)
        perplexity = perplexity / len(valid_data)

    # Print statistics
    print('Epoch ' + str(epoch + 1) + str(':'))
    print('Training loss : ' + str(training_loss))
    print('Validation loss : ' + str(validation_loss))
    print('Validation perplexity : ' + str(perplexity))

    # Save current model to a file
    model.__module__ = 'module'
    pickle.dump(model, open('model/epoch_' + str(epoch + 1) + '.pkl','wb'))
