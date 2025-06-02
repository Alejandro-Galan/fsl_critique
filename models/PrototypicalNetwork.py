import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class PrototypicalNetwork(nn.Module):
    # def __init__(self, keep_prob, XSupp, YSupp, encoder_features, \
    #              batch_size=100, num_channels=1, learning_rate=0.001, fce=False, num_classes_per_set=5, \
    #              num_samples_per_class=1, nClasses = 0, image_size = 28, best_accuracy = 0.0,
    #              x_dim=1, hid_dim=64, z_dim=64):    
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(PrototypicalNetwork, self).__init__()

        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
    
    def get_outputs(x_target, y_target, x_support_set, y_support_set, encoder):

        if len(x_target.shape) < 5:
            x_target = x_target.unsqueeze(1)
            y_target = y_target.unsqueeze(2)
        inputs_x = torch.cat((x_support_set, x_target), dim=1).cuda()

        if len(y_target.shape) == 1:
            y_target = y_target.unsqueeze(1)
        inputs_y = torch.cat((y_support_set, y_target.unsqueeze(2)), dim=1).cuda()

        # For each item
        outputs_x = []
        for i in range(inputs_x.shape[1]):
            outputs_x.append(encoder(inputs_x[:,i,:,:,:]))

        # inputs_y = inputs_y.squeeze(2).t().unsqueeze(2)
        return torch.stack(outputs_x,dim=1), inputs_y

