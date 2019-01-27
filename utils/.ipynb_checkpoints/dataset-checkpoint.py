import numpy as np
import glob
import utils.utils as util
import time
import os

class Dataset():
    def __init__(self, args):
        self.trainidx = []
        self.dataset_name = args.dataset_name
        self.path_to_annotations = os.path.join(args.annotation_path, args.dataset_name + '-Annotations/')
        self.path_to_features = os.path.join(args.I3D_path, self.dataset_name + '-I3D-JOINTFeatures.npy')
        self.labels = np.load(self.path_to_annotations + 'labels_all.npy')     # Specific to Thumos14
        self.classlist = np.load(self.path_to_annotations + 'classlist.npy')
        self.subset = np.load(self.path_to_annotations + 'subset.npy')
        self.testidx = []
        self.classwiseidx = []
        self.train_test_idx()
        self.currenttestidx = 0
        self.t_max = args.max_seqlen
        self.num_class = args.num_class
        self.classwise_feature_mapping()
        self.batch_size = args.batch_size
        self.feature_size = args.feature_size
        self.features = np.load(self.path_to_features, encoding='bytes')
        self.segments = np.load(self.path_to_annotations + 'segments.npy')
        self.labels_multihot = [util.strlist2multihot(labs,self.classlist) for labs in self.labels]


    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode('utf-8') == 'validation':   # Specific to Thumos14
                self.trainidx.append(i)
            else:
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode('utf-8'):
                        idx.append(i); break;
            self.classwiseidx.append(idx)


    def load_data(self, n_similar=3, is_training=True):
        if is_training==True:
            features = []
            labels = []
            idx = []

            # Load rest pairs
            rand_sampleid = np.random.choice(len(self.trainidx), size=self.batch_size)
            for r in rand_sampleid:
                idx.append(self.trainidx[r])

            feats = np.array([util.process_feat(self.features[i], self.t_max) for i in idx])
            labs = np.array([self.labels_multihot[i] for i in idx])
            
            return feats, labs

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]

            if self.currenttestidx == len(self.testidx)-1:
                done = True; self.currenttestidx = 0
            else:
                done = False; self.currenttestidx += 1
         
            return np.array(feat), np.array(labs), done
