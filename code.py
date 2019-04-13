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
    # start_time = time.time()
    for step in tqdm(range(0, len(data)//batch_size + 1)):
        offset = (step * batch_size) % (len(data) - batch_size)
        batch_data = data[offset:(offset + batch_size)]
    # Perform one training pass over the entire training set
    # for line in tqdm(data):
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
    # end_time = time.time()

    # # Calculate the perplexity on the validation set
    # with torch.no_grad():
    #     if torch.cuda.is_available():
    #         perplexity = torch.cuda.FloatTensor(len(valid_data))
    #     else:
    #         perplexity = torch.FloatTensor(len(valid_data))
    #     ind = 0
    #     for line in valid_data:
    #         if torch.cuda.is_available():
    #             char_embed = line[0].cuda()
    #             word_embed = line[1].cuda()
    #             h_ = torch.zeros(1, 1, 100).cuda()
    #             c_ = torch.zeros(1, 1, 100).cuda()
    #             valid_loss = torch.cuda.FloatTensor(char_embed.size(0)-1)
    #         else:
    #             char_embed = line[0]
    #             word_embed = line[1]
    #             h_ = torch.zeros(1, 1, 100)
    #             c_ = torch.zeros(1, 1, 100)
    #             valid_loss = torch.FloatTensor(char_embed.size(0)-1)
    #         for i in range(0, char_embed.size(0)-1):
    #             y_, (h_, c_) = model(char_embed[i], (h_, c_))
    #             valid_loss[i] = criterion(y_.view(1, 10000), word_embed[i+1].view(1))
    #         perplexity[ind] = torch.exp(torch.mean(valid_loss))
    #         ind += 1
    #     perplexity_avg = torch.mean(perplexity)

    # Print statistics
    print('Epoch ' + str(epoch + 1) + str(':'))
    # print('Time for epoch : ' + str(end_time - start_time))
    print('Training loss(last example) : ' + str(train_loss))
    # print('Validation perplexity : ' + str(perplexity_avg))

    # Save current model to a file
    pickle.dump(model, open('model/epoch_' + str(epoch + 1) + '.pkl','wb'))
