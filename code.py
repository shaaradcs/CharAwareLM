import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import time
from tqdm import tqdm


class LanguageModel(nn.Module):
    def __init__(self, char_vocab_size=55, input_dim_1=15, input_dim_2=32, kernel_width=5, hidden_dim=100, layer_dim=1, output_dim=10001):
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
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=layer_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)

    # def forward(self, x):
    #     # x : Sequence of indices corresponding to characters of a word
    #     # h : Tuple containing values of hidden and cell state of LSTM
    #     emb = self.embedding(x)
    #     h = torch.zeros(1, 1, self.hidden_dim).cuda()
    #     c = torch.zeros(1, 1, self.hidden_dim).cuda()
    #     results = torch.FloatTensor(x.shape[0], self.output_dim).cuda()
    #     for i in range(x.shape[0]):
    #         out = self.cnn(emb[i].view(1, 1, self.input_dim_1 * self.input_dim_2))
    #         out = self.tanh(out)
    #         out = self.maxpool(out).view(1, 1, self.hidden_dim)
    #         out, (h, c) = self.lstm(out, (h, c))
    #         results[i] = self.readout(out)
    #     return results

    def forward(self, x):
        # x : Sequence of indices corresponding to characters of a word
        # h : Tuple containing values of hidden and cell state of LSTM
        emb = self.embedding(x).permute(1,0,2,3)
        cnn_outs = torch.FloatTensor(x.shape[1], x.shape[0], self.hidden_dim).cuda()
        for i in range(x.shape[1]):
            out = self.cnn(emb[i].view(x.shape[0], 1, self.input_dim_1 * self.input_dim_2))
            out = self.tanh(out)
            cnn_outs[i] = self.maxpool(out).view(x.shape[0], self.hidden_dim)
        cnn_outs = cnn_outs.permute(1, 0, 2)
        h = torch.zeros(1, x.shape[1], self.hidden_dim).cuda()
        c = torch.zeros(1, x.shape[1], self.hidden_dim).cuda()
        out, (h, c) = self.lstm(cnn_outs, (h, c))
        results = self.readout(out)
        return results

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
model = LanguageModel()

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

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

batch_size = 128
# Load model
# model = pickle.load(open('model/epoch_11.pkl', 'rb'))

# for epoch in range(0, 30):
for epoch in range(0, 30):
    training_loss = torch.zeros(1).float().cuda()
    for step in tqdm(range(0, len(data)//batch_size + 1)):
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
        results = model(char_index)
        for i in range(batch_size):
            train_loss += criterion(results[i], word_index[i])
        train_loss.backward()
        optim.step()
        training_loss += train_loss
    training_loss = training_loss / (len(data) // batch_size)

    # Calculate the perplexity on the validation set
    with torch.no_grad():
        if torch.cuda.is_available():
            perplexity = torch.zeros(1).float().cuda()
        else:
            perplexity = torch.zeros(1).float()
        for line in valid_data:
            char_index = line[0].view(1, line[0].shape[0], -1).cuda()
            word_index = line[1].cuda()
            result = model(char_index)
            perplexity += torch.exp(criterion(result[0], word_index))
        perplexity = perplexity / len(valid_data)

    # Print statistics
    print('Epoch ' + str(epoch + 1) + str(':'))
    print('Training loss : ' + str(training_loss))
    print('Validation perplexity : ' + str(perplexity))

    # Save current model to a file
    pickle.dump(model, open('model/epoch_' + str(epoch + 1) + '.pkl','wb'))
