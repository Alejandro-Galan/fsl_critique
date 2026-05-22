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
    
    @staticmethod
    def _as_label_column(labels):
        """Return labels with shape [batch, sequence, 1].

        Prototypical loss expects integer class ids for both support and query
        samples. Some callers provide support labels as one-hot tensors for the
        MatchingNetwork path, so convert them back to class ids here before
        concatenating with target labels.
        """
        if labels.dim() == 3 and labels.size(-1) > 1:
            labels = labels.argmax(dim=-1)
        elif labels.dim() == 3 and labels.size(-1) == 1:
            return labels.long()
        elif labels.dim() == 1:
            labels = labels.unsqueeze(1)

        if labels.dim() != 2:
            raise ValueError(f"Expected labels with 1, 2 or 3 dims, got shape {tuple(labels.shape)}")

        return labels.unsqueeze(2).long()

    @staticmethod
    def get_outputs(x_target, y_target, x_support_set, y_support_set, encoder):

        if len(x_target.shape) < 5:
            x_target = x_target.unsqueeze(1)
        inputs_x = torch.cat((x_support_set, x_target), dim=1).cuda()

        support_labels = PrototypicalNetwork._as_label_column(y_support_set)
        target_labels = PrototypicalNetwork._as_label_column(y_target)
        inputs_y = torch.cat((support_labels, target_labels), dim=1).cuda()

        # For each item
        outputs_x = []
        for i in range(inputs_x.shape[1]):
            outputs_x.append(encoder(inputs_x[:, i, :, :, :]))

        # inputs_y = inputs_y.squeeze(2).t().unsqueeze(2)
        return torch.stack(outputs_x, dim=1), inputs_y

