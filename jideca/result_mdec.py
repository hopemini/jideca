import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
from models.model import DEC
import numpy as np
import math
import pickle
import os
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import argparse
import glob
import json

par = argparse.ArgumentParser()
par.add_argument("-t", "--types", default="semantic", choices=["real", "semantic_annotations"],
                 type=str, help="Choose a data type. (real/semantic_annotations)")
par.add_argument("-c", "--number_of_classes", default=34,
                 type=int, help="number of classes")
par.add_argument("-gid", "--gpu_id", default="0",
                 type=str, help="GPU_id")
par.add_argument("-js", "--jsd", action='store_true', help="use js-divergence")

par.add_argument("-beta", "--beta_reconstruction", default=0.1,
                 type=float, help="Hyperparameter for reconstruction term.")
par.add_argument("-gamma", "--gamma_alignment", default=0.1,
                 type=float, help="Hyperparameter for alignment term.")
par.add_argument("-sl", "--saved_log", default="./log",
                 type=str, help="log path")

par.add_argument("-i", "--ith", default=0,
                 type=int, help="ith iteration ")

args = par.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

## Set the parameters and data path
types = args.types
num_classes = args.number_of_classes
beta = args.beta_reconstruction
gamma = args.gamma_alignment
saved_log_path = args.saved_log
batch_size = 2400

# ../saved/log_mdec_re
# ../saved/log_midecseq2seq_re_23
# ../saved/log_dec_dnn
if saved_log_path != './log':
    _saved_log_path = saved_log_path.split('/')[-1].split('_')
    if _saved_log_path[-1] == '23':
        num_classes = 23
        if _saved_log_path[-2] == 're':
            types = 'real'
        elif _saved_log_path[-2] == 'se':
            types = 'semantic_annotations'
        else:
            types = 'dnn'
        type_name = _saved_log_path[-2]
    else:
        if _saved_log_path[-1] == 're':
            types = 'real'
        elif _saved_log_path[-1] == 'se':
            types = 'semantic_annotations'
        else:
            types = 'dnn'
        type_name = _saved_log_path[-1]
    data_type = _saved_log_path[1]
    
## jideca_b01r01re_34
data_name = "jideca_" + data_type + "_" + type_name + "_" + str(num_classes)

## Set each path
train_path = "./data/train_" + ("re" if types == "real" else "se") + ".pkl"
test_path = "./data/test_" + ("re" if types == "real" else "se") + "_" + str(num_classes) + ".pkl"


## Set data loader
root_path = "../data_processing/"

with open(train_path, "rb") as fr:
    img_train_loader = pickle.load(fr)

with open(test_path, "rb") as fr:
    img_test_loader = pickle.load(fr)

img_data_test_path = root_path + ("activity" if types == "real" else "semantic_annotations")
img_data_test_path = img_data_test_path + "/image/test_" + str(num_classes)

img_test_dataset = datasets.ImageFolder(root=img_data_test_path,
                           transform=transforms.Compose([
                               transforms.Resize((256,128)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5,0.5, 0.5)),
                           ]))

w2v_train_path = "data/word2vec_all/word2vec_all_data.npy"
w2v_test_path = "data/word2vec_" + str(num_classes) + "/word2vec_" + str(num_classes) + "_data.npy"
w2v_test_name_path = "data/word2vec_" + str(num_classes) + "/word2vec_" + str(num_classes) + "_names.json"

w2v_train_data = np.load(w2v_train_path)
w2v_test_data = np.load(w2v_test_path)
tensor_w2v_train_data = torch.Tensor(w2v_train_data)
tensor_w2v_test_data = torch.Tensor(w2v_test_data)

w2v_train_temp = torch.Tensor(w2v_train_data)
w2v_train_dataset = torch.utils.data.TensorDataset(tensor_w2v_train_data, w2v_train_temp)
text_train_loader = torch.utils.data.DataLoader(w2v_train_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=64)

text_test_names = json.loads(open(w2v_test_name_path).read())
text_test_names = text_test_names['name']


## Controlling the training process of DEC
class Trainer:
    def __init__(self, n_clusters, alpha, b, g):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = b
        self.gamma = g
        
    def save_result(self, name_list, classes, k):
        result_path = "../clustering/result_" + str(args.ith) + "/"
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        result_path = result_path + data_name + "/"
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        k_result = dict()
        for i in range(len(classes)):
            if str(classes[i]) not in k_result.keys():
                k_result[str(classes[i])] = []
            k_result[str(classes[i])].append(name_list[i])

        for k in k_result.keys():
            with open(result_path+str(int(k)+1), 'w') as f:
                for i in k_result[k]:
                    f.write(i+"\n")

    def result(self,
               img_train_loader,
               img_test_dataset,
               text_train_loader,
               w2v_test_data,
               text_test_names):

        if args.jsd:
            _log_path = (saved_log_path + "/pth/best_"
                               + types + "_idec_jsd_conv_dnn_save_model.pth")
        else:
            _log_path = (saved_log_path + "/pth/best_"
                               + types + "_idec_conv_dnn_save_model.pth")

        model = DEC(self.n_clusters)
        model.load_state_dict(torch.load(_log_path))
        model.eval()
        print('Initializing cluster center with pre-trained weights')
        mbk = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=batch_size)

        for img_data, text_data in tqdm(zip(img_train_loader, text_train_loader)):
            img, _ = img_data
            img = Variable(img).cuda()
            var, _ = text_data
            var = Variable(var).cuda()
            model.clustering(mbk, img, var, model)

        classes = []
        for data, img_path in tqdm(zip(img_test_dataset, img_test_dataset.imgs)):
            img = Variable(data[0]).cuda().unsqueeze(0)
            label = img_path[0].split('/')[-1].split('.')[0]
            text_d = w2v_test_data[int(text_test_names.index(label))]
            text = torch.from_numpy(text_d)
            var = text.cuda().unsqueeze(0)

            q, r , p, cnn_e, cnn_d, dnn_e, dnn_d = model(img, var)
            classes.append(torch.argmax(p, 1).item())

        self.save_result(text_test_names, classes, num_classes)

if __name__ == "__main__":
    trainer = Trainer(num_classes, 1.0, beta, gamma)
    
    trainer.result(img_train_loader,
                   img_test_dataset,
                   text_train_loader,
                   w2v_test_data,
                   text_test_names)
