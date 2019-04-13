import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import time


class LanguageModel(nn.Module):
    def __init__(self, char_vocab_size=54, input_dim_1=15, input_dim_2=32, kernel_width=5, hidden_dim=100, layer_dim=1, output_dim=10000):
        super(LanguageModel, self).__init__()

        self.char_vocab_size = char_vocab_size
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.kernel_width = kernel_width
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(char_vocab_size, input_dim_1)
        self.cnn = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=(kernel_width * input_dim_1), stride=input_dim_1)
        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool1d(kernel_size=(input_dim_2 - kernel_width + 1), padding=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, layer_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h):
        # x : Sequence of indices corresponding to characters of a word
        # h : Tuple containing values of hidden and cell state of LSTM
        out = self.embedding(x).view(1, 1, self.input_dim_1 * self.input_dim_2)
        out = self.cnn(out)
        out = self.tanh(out)
        out = self.maxpool(out).view(1, 1, self.hidden_dim)
        out, h = self.lstm(out, h)
        out = self.readout(out)
        return out, h

if torch.cuda.is_available():
    print('GPU available')
    print('Current(default) GPU device : ' + str(torch.cuda.current_device()))
    device = 1
    torch.cuda.set_device(device)
    print('GPU device set to ' + str(torch.cuda.current_device()))
else:
    print('GPU unavailable. Using CPU')

# Initialize model
model = LanguageModel()

if torch.cuda.is_available():
    model.cuda()

# Loading training data in the form of list of tensor tuples
data = pickle.load(open('data/train.txt.pkl', 'rb'))
print('Training data loaded')

# valid_data = pickle.load(open('data/valid.txt.pkl', 'rb'))
valid_data = pickle.load(open('data/valid.txt.pkl', 'rb'))
print('Validation data loaded')

learning_rate = 1

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load model
# model = pickle.load(open('model/epoch_11.pkl', 'rb'))

# for epoch in range(0, 30):
for epoch in range(0, 30):
    
    # Perform one training pass over the entire training set
    start_time = time.time()
    for line in data:
        if torch.cuda.is_available():
            char_embed = line[0].cuda().detach()
            word_embed = line[1].cuda().detach()
            h = torch.zeros(1, 1, 100, requires_grad=True).cuda()
            c = torch.zeros(1, 1, 100, requires_grad=True).cuda()
            train_loss = torch.zeros(1, requires_grad=True).cuda()
        else:
            char_embed = line[0].detach()
            word_embed = line[1].detach()
            h = torch.zeros(1, 1, 100, requires_grad=True)
            c = torch.zeros(1, 1, 100, requires_grad=True)
            train_loss = torch.zeros(1, requires_grad=True)
        optim.zero_grad()
        for i in range(0, char_embed.size(0)-1):
            y, (h, c) = model(char_embed[i], (h, c))
            train_loss += criterion(y.view(1, 10000), word_embed[i+1].view(1))
        train_loss.backward()
        optim.step()
    end_time = time.time()
    
    # Calculate the perplexity on the validation set
    if torch.cuda.is_available():
        perplexity = torch.cuda.FloatTensor(len(valid_data), requires_grad=False)
    else:
        perplexity = torch.FloatTensor(len(valid_data), requires_grad=False)
    ind = 0
    for line in valid_data:
        if torch.cuda.is_available():
            char_embed = line[0].cuda().detach()
            word_embed = line[1].cuda().detach()
            h = torch.zeros(1, 1, 100, requires_grad=False).cuda()
            c = torch.zeros(1, 1, 100, requires_grad=False).cuda()
            valid_loss = torch.cuda.FloatTensor(char_embed.size(0), requires_grad=False)
        else:
            char_embed = line[0].detach()
            word_embed = line[1].detach()
            h = torch.zeros(1, 1, 100, requires_grad=False)
            c = torch.zeros(1, 1, 100, requires_grad=False)
            valid_loss = torch.FloatTensor(char_embed.size(0)-1, requires_grad=False)
        for i in range(0, char_embed.size(0)-1):
            y, (h, c) = model(char_embed[i], (h, c))
            valid_loss[i] = criterion(y.view(1, 10000), word_embed[i+1].view(1))
        perplexity[ind] = torch.exp(torch.mean(valid_loss))
        ind += 1
    perplexity_avg = torch.mean(perplexity)

    # Print statistics
    print('Epoch ' + str(epoch + 1) + str(':'))
    print('Time for epoch : ' + str(end_time - start_time))
    print('Training loss(last example) : ' + str(train_loss))
    print('Validation perplexity : ' + str(perplexity_avg))

    # Save current model to a file
    pickle.dump(model, open('model/epoch_' + str(epoch + 1) + '.pkl','wb'))

