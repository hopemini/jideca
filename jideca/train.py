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
par.add_argument("-e", "--number_of_epochs", default=5000,
                 type=int, help="number of epochs")
par.add_argument("-c", "--number_of_classes", default=34,
                 type=int, help="number of classes")
par.add_argument("-gid", "--gpu_id", default="0",
                 type=str, help="GPU_id")
par.add_argument("-r", "--resume", action='store_true', help="use resume")
par.add_argument("-js", "--jsd", action='store_true', help="use js-divergence")

par.add_argument("--weight_decay", default=0.01, type=float,
                 help="Weight decay if we apply some.")
par.add_argument("-lr", "--learning_rate", default=0.001,
                 type=float, help="The initial learning rate for Adam.")
par.add_argument("--adam_epsilon", default=1e-8, type=float,
                 help="Epsilon for Adam optimizer.")
par.add_argument('--gradient_accumulation_steps', type=int, default=1,
                 help="Number of updates steps to accumualte before performing a backward/update pass.")
par.add_argument("--off_scheduling", action='store_false',
                 help="off_scheduling")

par.add_argument("-beta", "--beta_reconstruction", default=0.1,
                 type=float, help="Hyperparameter for reconstruction term.")
par.add_argument("-gamma", "--gamma_alignment", default=0.1,
                 type=float, help="Hyperparameter for alignment term.")
par.add_argument("-l", "--lambda_parameter", default=0.5,
                 type=float, help="lambda parameter")

par.add_argument("-p", "--save_per_epochs", default=5000,
                 type=int, help="save per epochs")

args = par.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

## Set the parameters and data path
types = args.types
num_epochs = args.number_of_epochs
num_classes = args.number_of_classes
learning_rate = args.learning_rate
beta = args.beta_reconstruction
gamma = args.gamma_alignment
lambda_param = args.lambda_parameter
save_per_epochs = args.save_per_epochs
batch_size = 2400

if abs(beta - 0) <= 1e-9:
    data_name = ("dec_" + ("re" if types == "real" else "se") + "_" + str(num_classes))
else:
    data_name = ("idec_" + ("re" if types == "real" else "se") + "_" + str(num_classes))

data_name += "_conv_dnn"
if args.jsd:
    data_name += "_jsd"

## Set each path
log_path = ("log/pth/best_loss_" + types + "_conv_autoencoder_save_model.pth")

train_path = "./data/train_" + ("re" if types == "real" else "se") + ".pkl"
test_path = "./data/test_" + ("re" if types == "real" else "se") + "_" + str(num_classes) + ".pkl"

if not os.path.exists("./log/"):
    os.mkdir("./log/")
if not os.path.exists("./log/pth/"):
    os.mkdir("./log/pth/")
save_log_path = "./log/check_point/"
save_img_path = "./log/img/"
if not os.path.exists(save_log_path):
    os.mkdir(save_log_path)
if not os.path.exists(save_img_path):
    os.mkdir(save_img_path)
save_log_path = (save_log_path + ("re" if types == "real" else "se")
        + "_" + str(num_classes) + "/")
save_img_path = (save_img_path + ("re" if types == "real" else "se")
        + "_" + str(num_classes) + "/")
if not os.path.exists(save_log_path):
    os.mkdir(save_log_path)
if not os.path.exists(save_img_path):
    os.mkdir(save_img_path)

## Set data loader
root_path = "../data_processing/"

with open(train_path, "rb") as fr:
    img_train_loader = pickle.load(fr)

with open(test_path, "rb") as fr:
    img_test_loader = pickle.load(fr)

img_data_test_path = root_path + ("activity" if types == "real" else types)
img_data_test_path = img_data_test_path + "/image/test_" + str(args.number_of_classes)

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


## Accuracy
nmi = normalized_mutual_info_score
def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

## Add noise
def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img

## Controlling the training process of DEC
class Trainer:
    def __init__(self, n_clusters, alpha, b, g):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = b
        self.gamma = g
        
    def logAccuracy(self,pred,label):
        print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
          % (acc(label, pred), nmi(label, pred)))

    def kld(self, p, q):
        res = torch.sum(p*torch.log(p/q),dim=-1)
        return res

    def jsd(self, q, r):
        m = 0.5 * (q + r)
        res = 0.5 * self.kld(q,m) + 0.5 * self.kld(q,m)
        return res

    #def validateOnCompleteTestData(self,test_loader,model):
    #    model.eval()
    #    for i,d in enumerate(test_loader):
    #        if i == 0:
    #            to_eval = model(d[0].cuda())[0].data.cpu().numpy()
    #            true_labels = d[1].cpu().numpy()
    #        else:
    #            to_eval = np.concatenate((to_eval, model(d[0].cuda())[0].data.cpu().numpy()), axis=0)
    #            true_labels = np.concatenate((true_labels, d[1].cpu().numpy()), axis=0)
    #    
    #    km = KMeans(n_clusters=len(np.unique(true_labels)))
    #    y_pred = km.fit_predict(to_eval)
    #    
    #    return acc(true_labels, y_pred), nmi(true_labels, y_pred)

    def train(self, img_train_loader, text_train_loader, num_epochs):
        if args.jsd:
            epoch_fmt = "_jsd_idec_epoch"
        else:
            epoch_fmt = "_idec_epoch"

        criterion = nn.MSELoss()
        # this method will start training for DEC cluster
        if args.resume:

            model = DEC(self.n_clusters, lambda_param)

            # get optimizer
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                eps=args.adam_epsilon,
                weight_decay=args.weight_decay
            )

            # learning rate scheduler
            scheduler = None
            if args.off_scheduling:
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=0.02,
                    epochs=num_epochs,
                    last_epoch=-1,
                    steps_per_epoch=int(len(img_train_loader)/args.gradient_accumulation_steps),
                    pct_start=0.3,
                    anneal_strategy="linear"
                )

            if args.jsd:
                _log_path = "./log/pth/" + types + "_idec_jsd_conv_dnn_save_model.pth"
            else:
                _log_path = "./log/pth/" + types + "_idec_conv_dnn_save_model.pth"
            model.load_state_dict(torch.load(_log_path))
            got_cluster_center = True
            pre_epochs = list()
            for e in glob.glob(save_log_path+'*'):
                pre_epochs.append(int(e.split('epoch')[-1]))
            pre_epochs.sort()
            start_e = pre_epochs[-1] + 1

            with open(save_log_path + epoch_fmt + str(start_e-1), 'r') as f:
                f_read = f.read()

            f_read_split = f_read.split('*')[-1].split(':')[-1]
            best_loss = float(f_read_split.split('[')[0])
            best_epoch = int(f_read_split.split('[')[-1].split('/')[0])

            print('Resume training..')
            print('Current epoch:[{}/{}], *best_loss:{}[{}/{}] '.format(
                                start_e, num_epochs, best_loss, best_epoch, num_epochs))

        else:
            best_epoch = 0
            best_loss = math.inf

            model = DEC(self.n_clusters, lambda_param)

            # get optimizer
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                eps=args.adam_epsilon,
                weight_decay=args.weight_decay
            )

            # learning rate scheduler
            scheduler = None
            if args.off_scheduling:
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=0.2,
                    epochs=num_epochs,
                    last_epoch=-1,
                    steps_per_epoch=int(len(img_train_loader)/args.gradient_accumulation_steps),
                    pct_start=0.3,
                    anneal_strategy="linear"
                )

            print('Initializing cluster center with pre-trained weights')
            mbk = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=batch_size)
            got_cluster_center = False
            start_e = 0

        if args.jsd:
            best_pt_path = "./log/pth/best_" + types + "_idec_jsd_conv_dnn_save_model.pth"
            pt_path = "./log/pth/" + types + "_idec_jsd_conv_dnn_save_model.pth"

        else:
            best_pt_path = "./log/pth/best_" + types + "_idec_conv_dnn_save_model.pth"
            pt_path = "./log/pth/" + types + "_idec_conv_dnn_save_model.pth"

        for epoch in range(start_e, num_epochs):
            for img_data, text_data in tqdm(zip(img_train_loader, text_train_loader)):
                img, _ = img_data
                img = Variable(img).cuda()
                var, _ = text_data
                var = Variable(var).cuda()

                # step 1 - get cluster center from batch
                # here we are using minibatch kmeans to be able to cope with larger dataset.
                if not got_cluster_center:
                    model.clustering(mbk, img, var, model)
                    if epoch > 1:
                        got_cluster_center = True
                else:
                    model.train()
                    # now we start training with acquired cluster center
                    q, r , p, cnn_e, cnn_d, dnn_e, dnn_d = model(img, var)

                    if lambda_param == 1:
                        r_loss = criterion(cnn_d, img)
                        kld_loss = self.kld(p,q).mean()
                    elif lambda_param == 0:
                        r_loss = criterion(dnn_d, var)
                        kld_loss = self.kld(p,r).mean()
                    else:
                        r_loss = criterion(cnn_d, img) + criterion(dnn_d, var)
                        kld_loss = self.kld(p,q).mean() + self.kld(p,r).mean()

                    # beta: reconstruction term
                    # gamma: alignment term
                    if args.jsd:
                        loss = kld_loss + (self.beta * r_loss)
                    else:
                        if abs(self.beta - 0) <= 1e-9:
                            # DEC (multimodal DEC)
                            loss = kld_loss
                        else:
                            # IDEC (multimodal IDEC)
                            loss = (self.beta * kld_loss) + r_loss

                    if args.jsd:
                        loss = loss + self.gamma * self.jsd(q,r).mean()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # schedule learning rate
                    if scheduler is not None:
                        scheduler.step()
            
            if got_cluster_center:
                #acc, nmi = self.validateOnCompleteTestData(test_loader,model)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_epoch = epoch+1
                    torch.save(model.state_dict(), best_pt_path)

                print('Epoch [{}/{}], loss:{:.4f}, *best_loss:{:.4f}[{}/{}]'
                      .format(epoch+1, num_epochs, loss.item(),
                          best_loss, best_epoch, num_epochs))
                with open(save_log_path + epoch_fmt + str(epoch+1), 'w') as f:
                    f.write("epoch [{}/{}], loss:{:.4f}, *best_nmi:{:.4f}[{}/{}]"
                      .format(epoch+1, num_epochs, loss.item(), best_loss, best_epoch, num_epochs))
                torch.save(model.state_dict(), pt_path)
                if epoch % save_per_epochs == 0:
                    torch.save(model.state_dict(), "./log/pth/" + types + "_jideca_" + str(epoch) + ".pth")

    def save_result(self, name_list, classes, k):
        result_path = "../clustering/result/"
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
            _log_path = ("./log/pth/best_"
                               + types + "_idec_jsd_conv_dnn_save_model.pth")
        else:
            _log_path = ("./log/pth/best_"
                               + types + "_idec_conv_dnn_save_model.pth")

        model = DEC(self.n_clusters, lambda_param)
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
    trainer.train(img_train_loader, text_train_loader, num_epochs)
    
    #trainer.result(img_train_loader,
    #               img_test_dataset,
    #               text_train_loader,
    #               w2v_test_data,
    #               text_test_names)
