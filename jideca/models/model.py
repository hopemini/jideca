import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class AutoEncoderConv(nn.Module):
    def __init__(self, num_classes):
        super(AutoEncoderConv, self).__init__()
        self.num_features = 64
        self.fc1 = nn.Linear(64*8*4, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 64)
        self.de_fc1 = nn.Linear(64, 256)
        self.de_fc2 = nn.Linear(256, 2048)
        self.de_fc3 = nn.Linear(2048, 64*8*4)

        self.encoder = nn.Sequential(
            # Input : 3*256*128
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 8*128*64

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 16*64*32

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 16*32*16

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 32*16*8

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 64*8*4
        )

        self.decoder = nn.Sequential(
            # 64*8*4

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(True),
            # 32*16*8

            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(True),
            # 16*32*16

            nn.ConvTranspose2d(16, 16, 2, stride=2),
            nn.ReLU(True),
            # 16*64*32

            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.ReLU(True),
            # 8*128*64

            nn.ConvTranspose2d(8, 3, 2, stride=2),
            nn.Sigmoid()
            # 3*256*128
        )

        self.alpha = 1.0
        self.clusterCenter = nn.Parameter(torch.zeros(num_classes, self.num_features))
        self.pretrainMode = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)

    def setPretrain(self,mode):
        self.pretrainMode = mode

    def updateClusterCenter(self, cc):
        self.clusterCenter.data = torch.from_numpy(cc)

    def getTDistribution(self, x, clusterCenter):
        xe = torch.unsqueeze(x,1).cuda() - clusterCenter.cuda()
        q = 1.0 / (1.0 + (torch.sum(torch.mul(xe,xe), 2) / self.alpha))
        q = q ** (self.alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, 1)).t() #due to divison, we need to transpose q
        return q

    def forward(self, x):
        # -- encoder --
        y = self.encoder(x)
        y = F.relu(self.fc1(y.view(y.size(0), -1)))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y_e = y

        # -- decoder --
        y = F.relu(self.de_fc1(y))
        y = F.relu(self.de_fc2(y))
        y = F.relu(self.de_fc3(y))
        y_d = self.decoder(y.view(y.size(0), 64, 8, 4))

        return y_e, self.getTDistribution(y_e, self.clusterCenter), y_d

## DNN autoencoder
class AutoEncoderDNN(nn.Module):
    def __init__(self, num_classes):
        super(AutoEncoderDNN, self).__init__()
        self.num_features = 64

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)

        self.de_fc1 = nn.Linear(64, 128)
        self.de_fc2 = nn.Linear(128, 256)

        self.alpha = 1.0
        self.clusterCenter = nn.Parameter(torch.zeros(num_classes, self.num_features))
        self.pretrainMode = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)

    def updateClusterCenter(self, cc):
        self.clusterCenter.data = torch.from_numpy(cc)

    def getTDistribution(self, x, clusterCenter):
        xe = torch.unsqueeze(x,1).cuda() - clusterCenter.cuda()
        q = 1.0 / (1.0 + (torch.sum(torch.mul(xe,xe), 2) / self.alpha))
        q = q ** (self.alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, 1)).t() #due to divison, we need to transpose q
        return q

    def forward(self, x):
        # -- encoder --
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y_e = y

        # -- decoder --
        y = F.relu(self.de_fc1(y))
        y_d = F.relu(self.de_fc2(y))

        return y_e, self.getTDistribution(y_e, self.clusterCenter), y_d

## DEC model for conv + dnn
class DEC(nn.Module):
    def __init__(self, n_clusters, _lambda=0.5):
        super(DEC, self).__init__()
        self._lambda = _lambda
        self.cnn_model = AutoEncoderConv(n_clusters).cuda()
        self.dnn_model = AutoEncoderDNN(n_clusters).cuda()

    def clustering(self, mbk, img, text, model):
        model.eval()
        cnn_ae, _, _ = self.cnn_model(img)
        cnn_ae = cnn_ae.data.cpu().numpy()
        cnn_pred = mbk.partial_fit(cnn_ae) # seems we can only get a centre from batch
        cluster_centers = mbk.cluster_centers_ # keep the cluster centers
        self.cnn_model.updateClusterCenter(cluster_centers)

        dnn_ae, _, _ = self.dnn_model(text)
        dnn_ae = dnn_ae.data.cpu().numpy()
        dnn_pred = mbk.partial_fit(dnn_ae) # seems we can only get a centre from batch
        cluster_centers = mbk.cluster_centers_ # keep the cluster centers
        self.dnn_model.updateClusterCenter(cluster_centers)

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return Variable((weight.t() / weight.sum(1)).t().data, requires_grad=True)

    def forward(self, img, var):
        cnn_e ,q, cnn_d = self.cnn_model(img)
        dnn_e ,r, dnn_d = self.dnn_model(var)
        #print("img q: {}".format(q.shape))
        #print("var r: {}".format(r.shape))
        if self._lambda == 1:
            p = self.target_distribution(q)
        elif self._lambda == 0:
            p = self.target_distribution(r)
        else:
            p = self._lambda * self.target_distribution(q) + (1 - self._lambda) * self.target_distribution(r)
        #print("mix p: {}".format(p.shape))

        return q, r , p, cnn_e, cnn_d, dnn_e, dnn_d
