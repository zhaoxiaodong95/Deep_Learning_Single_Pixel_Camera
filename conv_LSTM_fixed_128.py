import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as f

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

        self.cnn_before_lstm = nn.Sequential()
        self.cnn_before_lstm.add_module('conv0', nn.Conv2d(666, 256, 5, 1, 2))
        self.cnn_before_lstm.add_module('batchnorm0', nn.BatchNorm2d(256))
        self.cnn_before_lstm.add_module('relu0', nn.ReLU())
        self.cnn_before_lstm.add_module('conv1', nn.Conv2d(256, 128, 5, 1, 2))
        self.cnn_before_lstm.add_module('batchnorm1', nn.BatchNorm2d(128))

        self.cnn_after_lstm = nn.Sequential()
        self.cnn_after_lstm.add_module('conv2', nn.Conv2d(64, 32, 3, 1, 1))
        self.cnn_after_lstm.add_module('batchnorm2', nn.BatchNorm2d(32))
        self.cnn_after_lstm.add_module('relu2', nn.ReLU())
        self.cnn_after_lstm.add_module('conv3', nn.Conv2d(32, 1, 3, 1, 1))

        self.cnn_meas = nn.Sequential()
        self.cnn_meas.add_module('conv4', nn.Conv2d(1, 256, 3, 1, 1))
        self.cnn_meas.add_module('batchnorm4', nn.BatchNorm2d(256))
        self.cnn_meas.add_module('relu4', nn.ReLU())
        self.cnn_meas.add_module('conv5', nn.Conv2d(256, 666, 1, 1, 0))
        self.cnn_meas.add_module('batchnorm5', nn.BatchNorm2d(666, affine=False))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

        self._all_layers = []
        name = 'cell0'
        cell = ConvLSTMCell(128, 64, 3)
        setattr(self, name, cell)
        self._all_layers.append(cell)
        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, image, phi_0):
        internal_state = []
        batch_size, seq_len, width, height = image.size()
        output_seq = Variable(torch.zeros(batch_size, seq_len, width, height)).cuda()

        phi_t = f.normalize(phi_0, p=2, dim=1)
        for step in range(self.step):
            image_t = image[:, step, :]

            y_t = torch.sum(image_t[:, None, :, :] * phi_t, dim=(2, 3))
            x_1 = y_t[:, :, None, None] * phi_t
            x = self.cnn_before_lstm(x_1)

            for i in range(self.num_layers):  ## If you use more than one layer
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

            # phi_t = torch.sign(phi_0 + self.gamma * self.cnn_meas(x_output))
            # phi_t = f.normalize(phi_t, p=2, dim=1)

            output_seq[:, step, :] = x_output.squeeze(1)

        return output_seq, (x, new_c)


    def smoothclamp(self, x, mi, mx):
        return mi + (mx-mi)*(lambda t: torch.where(t < 0 , torch.cuda.FloatTensor(t.size()).fill_(0), torch.where( t <= 1 , 3*t**2-2*t**3, torch.cuda.FloatTensor(t.size()).fill_(1))))((x-mi)/(mx-mi))


    def clampoid(self, x, mi, mx):
        return mi + (mx-mi)*(lambda t: 0.5*(1+200**(-t+0.5))**(-1) + 0.5*torch.where(t < 0 , torch.cuda.FloatTensor(t.size()).fill_(0), torch.where( t <= 1 , 3*t**2-2*t**3, torch.cuda.FloatTensor(t.size()).fill_(1) ) ) )( (x-mi)/(mx-mi) )

