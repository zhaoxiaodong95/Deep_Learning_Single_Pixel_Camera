import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import normalize
import numpy as np



class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        # assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.relu(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.relu(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):

        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'

        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, frame_size, n_filters, hidden_size, device, step=1): #, effective_step=[1]
        super(ConvLSTM, self).__init__()
        self.frame_size = frame_size
        self.hidden_size = hidden_size
        self.n_filters = n_filters
        self.device = device
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step

        ## NOTE: kernel size can be [5, 1, 5, 9]. For adaptive part, kernel size should be 9. If I make it less than

        ks = [9, 1,  5, 9]  ## kernel size,, if kernel size is 1 pads
        ps = [4, 0,  2, 4]  ## Padding size
        ss = [1, 1,  1, 1]  ## Stride size
        nm = [128, 32, 1, int(self.n_filters)]  ## Number of output channels

        self.cnn_before_lstm = nn.Sequential()

        def convRelu_before_LSTM(i, Relu=True, batchNormalization=True, Maxpooling=False):
            nIn = 1 if i == 0 else nm[i - 1]
            nOut = nm[i]
            self.cnn_before_lstm.add_module('conv{0}'.format(i),
                                            nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                self.cnn_before_lstm.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))

            # if leakyRelu:
            #     self.cnn.add_module('relu{0}'.format(i),
            #                    nn.LeakyReLU(0.2, inplace=True))
            if Relu:
                self.cnn_before_lstm.add_module('relu{0}'.format(i), nn.ReLU())
            else:
                self.cnn_before_lstm.add_module('relu{i}'.format(i), nn.Tanh())

            if Maxpooling:
                self.cnn_before_lstm.add_module('maxpool{0}'.format(i), nn.MaxPool2d(2, 2, padding=1))

        self.cnn_after_lstm = nn.Sequential()

        def convRelu_after_LSTM(i, Relu=True, batchNormalization=True, Maxpooling=True):
            nIn = self.hidden_channels[-1] if i==1 else nm[i-1]
            nOut = nm[i]
            self.cnn_after_lstm.add_module('conv{0}'.format(i),
                                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                self.cnn_after_lstm.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))

            # if leakyRelu:
            #     self.cnn.add_module('relu{0}'.format(i),
            #                    nn.LeakyReLU(0.2, inplace=True))
            if Relu:
                self.cnn_after_lstm.add_module('relu{0}'.format(i), nn.ReLU())
            # if not Relu:
            #     self.cnn_after_lstm.add_module('relu{0}'.format(i), nn.Sigmoid())



        convRelu_before_LSTM(0)

        convRelu_after_LSTM(1)             # 32x frame size
        convRelu_after_LSTM(2, Relu=False) # 1x frame size

        self.middle_layer = nn.Linear(int(self.n_filters), 1 * self.frame_size[0] * self.frame_size[1], bias=True)
        self.bn_out_1st = nn.BatchNorm1d(self.frame_size[0] * self.frame_size[1])
        ## weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.5)

        # self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size[i])
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, image, phi_t):
        internal_state = []
        batch_size, seq_len, width, height = image.size()

        output_seq = Variable(torch.zeros(batch_size, seq_len, width, height)).cuda(non_blocking=True)

        for step in range(self.step):
            image_t = image[:, step, :]

            ## Measurement Matrix
            y_t = torch.sum(torch.sum(torch.mul(image_t.view(batch_size, 1, width, height), phi_t), 2), 2)   ## measurement vector

            image_mid = torch.relu(self.bn_out_1st(self.middle_layer(y_t)))                                    ## 1st Linear Layer
            x_1 = image_mid.view(batch_size, 1, self.frame_size[0], self.frame_size[1])                        ## Resize for Conv-LSTM
            x = self.cnn_before_lstm(x_1)


            ## If you use more than one LSTM layer
            for i in range(self.num_layers):

                # All cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # Do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)

            x_output = self.cnn_after_lstm(x)
            output_seq[:, step, :] = x_output.squeeze(1)

        return output_seq

