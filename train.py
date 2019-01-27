import os
import torch.nn as nn
from utils.utils import *
from utils.args import Args
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from utils.dataset import Dataset
from model.attention_module import Model
from utils.loss import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def main():

    # Category to sentence
    class_name = {0: ["baseball pitch", "throw a baseball", "baseball throw"],
                  1: ["basketball dunk", "dunk a basketball", "slam dunk basketball"],
                  2: ["billiards"],
                  3: ["clean and jerk", "weight lifting movement"],
                  4: ["cliff diving", "high diving", "diving"],
                  5: ["cricket shot"],
                  6: ["cricket bowling", "cricket movement", "bowl cricket"],
                  7: ["diving", "jumping into water", "falling into water"],
                  8: ["frisbee catch", "catch frisbee"],
                  9: ["golf swing", "golf stroke"],
                  10: ["hammer throw", "throw a hammer"],
                  11: ["high jump"],
                  12: ["javelin throw", "throw a spear"],
                  13: ["long jump", "jump contest"],
                  14: ["pole vault", "a person uses a long flexible pole to jump over a bar"],
                  15: ["shot put"],
                  16: ["soccer penalty"],
                  17: ["tennis swing"],
                  18: ["throw discus", "discus"],
                  19: ["volleyball spiking", "volleyball", ]}

    # Initialize the arguments
    args = Args()
    checkpoint_model_name = args.model_name

    # Specify GPU
    device = torch.device(args.gpu)
    args.device = device

    # Initialize dataset
    dataset = Dataset(args)

    # Duo Model testing
    lstm_input_size = 300
    hidden_dim = 300
    output_dim = 100
    batch_size = 1
    num_layers = 2

    model = Model(lstm_input_size, hidden_dim, batch_size=batch_size,
                  time_steps=8, args=args, output_dim=output_dim, num_layers=num_layers)

    # Load pre-trained classification network for bootstrapping
    checkpoint = torch.load(args.t_cam)
    pretrained_dict = checkpoint['state_dict']
    fc_weight = pretrained_dict['fc3.weight']

    # filter out unnecessary keys and load valid params
    # model_dict = model.state_dict()
    # checkpoint = torch.load(args.t_cam)
    # pretrained_dict = checkpoint['state_dict']
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    # filter out unnecessary keys and load valid params
    model_dict = model.visual_model.state_dict()
    checkpoint = torch.load(args.t_cam)
    pretrained_dict = checkpoint['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.visual_model.load_state_dict(model_dict)

    model.to(device)
    model.textual_model.to(device)
    model.visual_model.to(device)
    print('model created')

    # Store the fc weights for t-cam prediction
    # args.weights = model.visual_model.fc3.weight.detach().cpu()
    args.weights = fc_weight.cpu()
    args.tao = 0.9
    args.lr = 0.01

    # Loss defined here
    marginrankingloss = nn.MarginRankingLoss(0.1)
    adaptive_margin_loss = Adaptive_Margin_Loss()
    l1_regu = nn.L1Loss(size_average=False)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    writer = SummaryWriter()

    for epoch in range(args.max_iter):
        # Randomly extract 10 video clips' I3D feature
        features, labels = dataset.load_data()

        # Features are aligned in 750 frames all the same, now trunk it into max length
        seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
        features = features[:, :np.max(seq_len), :]

        # Convert to CUDA tensor
        features = torch.from_numpy(features).float().to(device)
        labels = torch.from_numpy(labels).float().to(device)

        # Generate texts from categories
        text_list, labels = one_label_text(class_name, labels)

        # Visualize T-CAM result for comparison
        clses = [[idx for idx, cls in enumerate(label) if cls == 1.] for label in labels]
        t_proposals = temporal_proposals(args.weights, features.detach().cpu(), clses)

        # Predict
        attention_weights, visual_feature, textual_feature, pos_feature, neg_feature, neg_features_mean \
            = model(features, text_list, None)

        # L1 regularization on attention weights
        # l1_loss = 0
        # for attention in attention_weights:
        #     target = torch.zeros_like(attention)
        #     l1_loss += l1_regu(attention, target)

        # loss = adaptive_margin_loss(pos_feature, neg_feature, textual_feature,  args.device)\
        #        +euclidean_distance(visual_feature, textual_feature)

        # Squared Loss, Margin Ranking Loss
        pos_distance = euclidean_distance(pos_feature, textual_feature, 1)
        neg_distance = euclidean_distance(neg_features_mean, textual_feature, 1)
        target = torch.zeros(visual_feature.shape[0]).cuda() - 1
        margin_loss = marginrankingloss(pos_distance, neg_distance, target)
        loss = adaptive_margin_loss(pos_feature, neg_feature, textual_feature, args.device) + 0.01 * margin_loss

        # Back Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save checkpoint
        if epoch % 3000 == 0 and epoch is not 0:
            # Reduce lr
            args.tao /= 3
            args.lr /= 1.5
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

            # Checkpoint structure
            model_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            directory = args.checkpoint_path + checkpoint_model_name + "/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(model_state,
                       os.path.join(directory + 'epoch_{:03}.pth'.format(epoch)))

        # Print out training loss
        loss_value = loss.detach().cpu().tolist()
        writer.add_scalar('runs/', loss_value, epoch)

        if epoch % 20 == 0:
            print('Epoch:{:03}, Loss: {:02}'.format(epoch, loss_value))
            # Display attention weights and loss value
            plt.plot(attention_weights[0].tolist(), c='b')
            plt.plot(t_proposals[0], c='r')
            directory = "./visualization/" + checkpoint_model_name + "/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(directory + str(epoch) + ".png")
            plt.clf()
            plt.close("all")

    writer.export_scalars_to_json(args.json)
    writer.close()

if __name__ == "__main__":
    main()

