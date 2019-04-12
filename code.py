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

        print('Initialized module')

    def forward(self, x, h):
        # x : Sequence of indices corresponding to characters of a word
        out = self.embedding(x).view(1, 1, self.input_dim_1 * self.input_dim_2)
        out = self.cnn(out)
        out = self.tanh(out)
        out = self.maxpool(out).view(1, 1, self.hidden_dim)
        out, h = self.lstm(out, h)
        out = self.readout(out)
        return out, h

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    torch.cuda.set_device(1)


# Initialize model
model = LanguageModel()

if torch.cuda.is_available():
    model.cuda()

# Loading training data in the form of list of tensor tuples
data = pickle.load(open('data/train.txt.pkl','rb'))
print('Data loaded')

learning_rate = 0.05

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(0, 30):
    start_time = time.time()
    for line in data:
        if torch.cuda.is_available():
            char_embed = line[0].cuda()
            word_embed = line[1].cuda()
            h = torch.zeros(1, 1, 100).cuda()
            c = torch.zeros(1, 1, 100).cuda()
            loss = torch.zeros(1).cuda()
        else:
            char_embed = line[0]
            word_embed = line[1]
            h = torch.zeros(1, 1, 100)
            c = torch.zeros(1, 1, 100)
            loss = torch.zeros(1)
        optim.zero_grad()
        for i in range(0, len(line)-1):
            y, (h, c) = model(char_embed[i], (h, c))
            loss += criterion(y.view(1, 10000), word_embed[i].view(1))
        loss.backward()
        optim.step()
    end_time = time.time()

    # Print statistics
    print('Time for epoch ' + str(epoch + 1) + ' : ' + str(end_time - start_time))
    print('Loss : ' + str(loss))

    # Save current model to a file
    pickle.dump(model, open('model/epoch_' + str(epoch + 1) + '.pkl','wb'))

