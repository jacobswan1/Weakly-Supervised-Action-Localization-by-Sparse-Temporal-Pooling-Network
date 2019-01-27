class Args:
    def __init__(self):
        self.lr = 0.0001
        self.dataset_name = 'Thumos14reduced'
        self.num_class = 20
        self.feature_size = 2048
        self.batch_size = 24
        self.max_seqlen = 750
        self.feature_dim = 2048
        self.model_name = 'weakloc'
        self.pretrained_ckpt = None
        self.max_iter = 50000
        self.checkpoint_path = './checkpoint/'
        self.annotation_path = './annotations/'
        self.I3D_path = './I3D_features/'
        self.json = './logs/all_scalars.json'
        self.path_to_glove = './checkpoint/glove.840B.300d.pkl'
        self.t_cam = './checkpoint/T-CAM.pth'
        self.pre_train = './checkpoint/LSTM-marginloss-topdown.pth'
        self.model_name = 'model_nlp_top-down_margin_loss=0.01_t-cam/'
        self.gpu = 'cuda:0'
        self.l1_weight = 0
