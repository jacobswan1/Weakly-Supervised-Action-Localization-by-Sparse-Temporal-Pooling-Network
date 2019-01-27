import torch.nn as nn
from model.language_encoder import Language_encoder
from model.visual_encoder import Visual_model


class Model(nn.Module):
    """Args:
        Video I3D features and raw natural text.
    """

    def __init__(self, input_dim, hidden_dim, batch_size, time_steps, args, output_dim=100,
                 num_layers=2, lstm_input_size=300):
        super(Model, self).__init__()
        """
            Create Duo Stream Model.
        """
        self.textual_model = Language_encoder(lstm_input_size, hidden_dim, batch_size=batch_size,
                                              time_steps=8, args=args, output_dim=output_dim, num_layers=num_layers)
        self.visual_model = Visual_model(args)

    def forward(self, visual_feature, text, t_proposals=None, test=False):
        """
            pos/neg_feature: (batch, 2048).
                mean_pooled representations.
        """
        textual_feature = self.textual_model(text)
        attention_weights, visual_feature, pos_feature, neg_feature, test_features = \
            self.visual_model(visual_feature, text, t_proposals=t_proposals, test=test)

        return attention_weights, visual_feature, textual_feature, pos_feature, neg_feature, test_features
