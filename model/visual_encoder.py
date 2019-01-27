# Load arguments
import torch
import torch.nn as nn
from model.language_encoder import Language_encoder
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Visual_model(nn.Module):
    """Args:
    feature_dim: dimension of the feature from I3D model.
    """

    def __init__(self, args):
        super(Visual_model, self).__init__()

        self.feature_dim = args.feature_dim
        self.fc0 = nn.Linear(args.feature_dim, 1024)
        self.fc1 = nn.Linear(1024, 256)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self.fc_lr = nn.Linear(2048, 100)
        self.language_encoder = Language_encoder(300, 300, batch_size=1, time_steps=8, args=args, output_dim=256, num_layers=2)

        # Bilinear Module: input: (# seg * 2048d), (1 * 300d); output: (# seg * 256)
        self.bilinear_pooling = nn.Bilinear(256, 100, 256)

    # Convert attention list to binary gates
    def attention_to_binary(self, attention_weights):
        result = []
        for attention in attention_weights:
            threshold = (torch.max(attention) - (torch.max(attention) - torch.min(attention)) * 0.5)
            result.append([1 if s > threshold else 0 for idx, s in enumerate(attention)])
        return result

    def forward(self, features_list, text_list, t_proposals=None, test=False):
        """Build the attention module.

        Args:
        features_list: (batch_size, num_frame, feat_depth)
        t_proposals:    temporal proposals generated from T-CAM, for boostrapping LSTM training.
        language_vector: top-down signal, (batch_size, feat_dim)

        Returns:
        The attention weights, weigted features
        """
        # Test_features are returning tensors (batch, length, feat_dim)
        attention_weights = []
        weighted_features = []
        pos_features = []
        neg_features = []
        neg_features_mean = []
        test_features = []
        text_features = self.language_encoder(text_list)

        # Iterate through batch since length of each video segment varies
        for idx, video_features in enumerate(features_list):
            # Trunk feature into real length
            seq_len = (torch.abs(video_features).max(dim=1)[0] > 0).sum().tolist()
            video_features = video_features[: seq_len, :]

            # Expand the size of language size
            language_feat = text_features[idx]
            language_feat = self.sigmoid(language_feat.expand(video_features.shape[0], language_feat.shape[-1]))

            # Iterate through video segments
            bilinear = self.relu(self.fc1(self.relu(self.fc0(video_features))) * language_feat)
            output = self.sigmoid(self.fc2(bilinear))
            #             output = gumbelSoftmax(self.fc2(bilinear), args.tao)

            # Temporal Pool
            weighted_pooling = (output * video_features).sum(0) / video_features.shape[0]

            # If testing, feed weighted video_features (no mean pool) to fc_lr
            if test:
                test_features.append(self.fc_lr((output * video_features)))

            # Save weights/features
            output = output.reshape(output.shape[0])
            attention_weights.append(output)
            weighted_features.append(weighted_pooling)

            # Pool pos/neg segments features from T-CAM proposals
            if t_proposals is not None:
                pos_list = [index for index, l in enumerate(t_proposals[idx]) if l == 1.]
                neg_list = [index for index, l in enumerate(t_proposals[idx]) if l == 0.]
                # Reduce mean over positive and negative video features
                pos_feature = torch.stack([feat for index, feat in enumerate(video_features)
                                           if index in pos_list]).sum(0) / len(pos_list)
                neg_feature = torch.stack([feat for index, feat in enumerate(video_features)
                                           if index in neg_list]).sum(0) / len(neg_list)

                pos_features.append(pos_feature)
                neg_features.append(neg_feature)
            # If provide no T-Proposals, we generate pos-neg through self attention
            else:
                # Retrieve pos-neg pairs according to threshold
                pos_neg = self.attention_to_binary(output.reshape((1, -1)))[0]
                pos_list = [index for index, l in enumerate(pos_neg) if l == 1.]
                neg_list = [index for index, l in enumerate(pos_neg) if l == 0.]
                pos_feature = torch.stack([feat for index, feat in enumerate(video_features)
                                           if index in pos_list]).sum(0) / len(pos_list)
                neg_feature = torch.stack([feat for index, feat in enumerate(video_features)
                                           if index in neg_list])
                neg_mean = torch.stack([feat for index, feat in enumerate(video_features)
                                           if index in neg_list]).sum(0) / len(neg_list)
                neg_features_mean.append(neg_mean)
                pos_features.append(pos_feature)
                neg_features.append(self.fc_lr(neg_feature))

        # Reshape to tensor
        weighted_features = torch.stack(weighted_features)

        # Feed temporal features to fc regression layer
        # output: aggregated visual representations
        output = self.fc_lr(weighted_features)

        if test:
            return attention_weights, output, None, None, test_features
        else:
            # training without t-cam bootstrapping
            if t_proposals is None:
                pos_features = torch.stack(pos_features)
                pos_features = self.fc_lr(pos_features)
                neg_features_mean = torch.stack(neg_features_mean)
                neg_features_mean = self.fc_lr(neg_features_mean)
                return attention_weights, output, pos_features, neg_features, neg_features_mean
            # training with t-cam bootstrapping
            else:
                pos_features = torch.stack(pos_features)
                neg_features = torch.stack(neg_features)
                pos_features = self.fc_lr(pos_features)
                neg_features = self.fc_lr(neg_features)
                return attention_weights, output, pos_features, neg_features, None
