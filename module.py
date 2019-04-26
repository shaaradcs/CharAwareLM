import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import time
from tqdm import tqdm


class LanguageModel(nn.Module):
    def __init__(self, char_vocab_size=55, input_dim_1=15, input_dim_2=32, kernel_width=5, hidden_dim=1000, lstm_hidden=300, layer_dim=2, output_dim=9977):
        super(LanguageModel, self).__init__()

        self.char_vocab_size = char_vocab_size
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.kernel_width = kernel_width
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(char_vocab_size, input_dim_1)

        self.cnn = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=(kernel_width * input_dim_1), stride=input_dim_1)
        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool1d(kernel_size=(input_dim_2 - kernel_width + 1), padding=0)
        self.transform_1 = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid_1 = nn.Sigmoid()
        self.highway_1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu_1 = nn.ReLU()
        self.lstm_forward_1 = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden, num_layers=1, dropout=0.5)
        self.lstm_backward_1 = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden, num_layers=1, dropout=0.5)
        self.lstm_forward_2 = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden, num_layers=1, dropout=0.5)
        self.lstm_backward_2 = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden, num_layers=1, dropout=0.5)
        self.dropout = nn.Dropout(p=0.5)
        self.readout = nn.Linear(lstm_hidden * 2, output_dim)

        """
        self.dropout_forward = nn.Dropout(p=0.5)
        self.dropout_backward = nn.Dropout(p=0.5)
        self.readout_forward = nn.Linear(lstm_hidden, output_dim)
        self.readout_backward = nn.Linear(lstm_hidden, output_dim)
        """

    def init_weights(self):
        initrange = 0.05
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.readout.weight.data.uniform_(-initrange, initrange)
        self.readout.bias.data.zero_()

        """
        self.readout_forward.weight.data.uniform_(-initrange, initrange)
        self.readout_forward.bias.data.zero_()
        self.readout_backward.weight.data.uniform_(-initrange, initrange)
        self.readout_backward.bias.data.zero_()
        """

    def forward(self, x):

        emb = self.embedding(x).permute(1,0,2,3)

        highway_outs = torch.FloatTensor(x.shape[1], x.shape[0], self.hidden_dim).cuda()
        for i in range(x.shape[1]):
            out = self.cnn(emb[i].view(x.shape[0], 1, self.input_dim_1 * self.input_dim_2))
            out = self.tanh(out)
            cnn_out = self.maxpool(out).view(x.shape[0], self.hidden_dim)
            transform_out = self.sigmoid_1(self.transform_1(cnn_out))
            highway_outs[i] = transform_out * self.relu_1(self.highway_1(cnn_out)) +  (1 - transform_out) * cnn_out

        out_forward, (h, c) = self.lstm_forward_1(highway_outs)
        out_backward, (h, c) = self.lstm_backward_1(highway_outs)

        # embed_1 = torch.cat((out_forward[:x.shape[1]-2], out_backward[2:]), dim=2).permute(1, 0, 2)

        out_forward, (h, c) = self.lstm_forward_2(out_forward)
        out_backward, (h, c) = self.lstm_backward_2(out_backward)

        embed_2  = torch.cat((out_forward[:x.shape[1]-2], out_backward[2:]), dim=2).permute(1, 0, 2)
        out = self.dropout(embed_2)
        result = self.readout(out)
        return result

        """
        out_forward = out_forward.permute(1, 0, 2)
        out_backward = out_backward.permute(1, 0, 2)

        out_forward = self.dropout_forward(out_forward)
        out_backward = self.dropout_backward(out_backward)

        results_f = self.readout_forward(out_forward)
        results_b = self.readout_backward(out_backward)

        return results_f, results_b
        """

    def embeddings(self, x):

        emb = self.embedding(x).permute(1,0,2,3)

        highway_outs = torch.FloatTensor(x.shape[1], x.shape[0], self.hidden_dim).cuda()
        for i in range(x.shape[1]):
            out = self.cnn(emb[i].view(x.shape[0], 1, self.input_dim_1 * self.input_dim_2))
            out = self.tanh(out)
            cnn_out = self.maxpool(out).view(x.shape[0], self.hidden_dim)
            transform_out = self.sigmoid_1(self.transform_1(cnn_out))
            highway_outs[i] = transform_out * self.relu_1(self.highway_1(cnn_out)) +  (1 - transform_out) * cnn_out

        out_forward, (h, c) = self.lstm_forward_1(highway_outs)
        out_backward, (h, c) = self.lstm_backward_1(highway_outs)

        embed_1 = torch.cat((out_forward, out_backward), dim=2).permute(1, 0, 2)

        out_forward, (h, c) = self.lstm_forward_2(out_forward)
        out_backward, (h, c) = self.lstm_backward_2(out_backward)

        embed_2  = torch.cat((out_forward, out_backward), dim=2).permute(1, 0, 2)

        return embed_1, embed_2
