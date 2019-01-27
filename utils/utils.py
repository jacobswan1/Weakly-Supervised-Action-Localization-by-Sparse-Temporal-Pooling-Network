import torch
import random
import numpy as np


def str2ind(categoryname,classlist):
   return [i for i in range(len(classlist)) if categoryname==classlist[i].decode('utf-8')][0]


def strlist2indlist(strlist, classlist):
	return [str2ind(s,classlist) for s in strlist]


def strlist2multihot(strlist, classlist):
    return np.sum(np.eye(len(classlist))[strlist2indlist(strlist,classlist)], axis=0)


def idx2multihot(id_list,num_class):
   return np.sum(np.eye(num_class)[id_list], axis=0)


def random_extract(feat, t_max):
   r = np.random.randint(len(feat)-t_max)
   return feat[r:r+t_max]


def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0,min_len-np.shape(feat)[0]), (0,0)), mode='constant', constant_values=0)
    else:
       return feat


def process_feat(feat, length):
    if len(feat) > length:
        return random_extract(feat, length)
    else:
        return pad(feat, length)


def write_to_file(dname, dmap, cmap, itr):
    fid = open(dname + '-results.log', 'a+')
    string_to_write = str(itr)
    for item in dmap:
        string_to_write += ' ' + '%.2f' %item
    string_to_write += ' ' + '%.2f' %cmap
    fid.write(string_to_write + '\n')
    fid.close()


# Randomly selecting out single label and cnovert it to text
def one_label_text(class_name, labels, test=False):
    if not test:
        tem_labels = torch.zeros_like(labels)
        target = [[idx for idx, item in enumerate(label.tolist()) if item == 1] for label in labels]
        for i, l in enumerate(target):
            idx = np.random.choice(l)
            tem_labels[i][idx] = 1
        text = label_to_text(class_name, tem_labels)
        return text, tem_labels
    else:
        target = [idx for idx, item in enumerate(labels.tolist()) if item == 1.]
        if len(target) == 0:
            print('Error here, not valid label assigned for this test case!')
            target.append(0)
        return_labels = torch.zeros(len(target), labels.shape[0])
        return_text = []
        for i, t in enumerate(target):
            return_labels[i][t] = 1
            text = label_to_text(class_name, [return_labels[i]])
            return_text.append(text)
        return return_text, return_labels


# Stacking all laebls to language format
def label_to_text(class_name, labels):
    text_list = []
    for label in labels:
        idxs = [idx for idx, val in enumerate(label) if val == 1]
        text = ''
        for idx in idxs:
            text += ' ' + random.choice(class_name[idx])
        text_list.append(text)
    return text_list


# T-cam with specific label
def temporal_proposals(weights, features, clses):
    """
        Return the binary temporal proposals based on T-CAM.
        Input:
            features:    (batch, # of segments, # of dim)
            cls:         (batch, class_label)
        Return:    [batch, # of untrimmed segments(binary)]
    """
    proposals = []
    # Iterate through batch
    for idx, video_feature in enumerate(features):
        # Chunk video feature to real length
        seq_len = (torch.abs(video_feature).max(dim=1)[0] > 0).sum().tolist()
        video_feature = video_feature[: seq_len, :]
        scores = np.zeros(seq_len)
        for seg_id, feature in enumerate(video_feature):
            for cls in clses[idx]:
                score = (feature * weights[cls]).sum()
                scores[seg_id] += score
        # Labeling segments with scores larger than threshold
        threshold = (np.max(scores) - (np.max(scores) - np.min(scores)) * 0.5)
        proposals.append([1 if s > threshold else 0 for idx, s in enumerate(scores)])
    return proposals