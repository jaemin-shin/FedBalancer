import torch
from torch import nn
from model import Model
import numpy as np
from utils.model_utils import build_net
from baseline_constants import ACCURACY_KEY

INPUT_SIZE = 28

class ClientModel(Model):
    def __init__(self,seed,dataset,model_name,lr,num_classes,cfg=None):
        self.seed=seed
        self.dataset=dataset
        self.model_name=model_name
        self.lr = lr
        self.num_classes=num_classes

        # TODO: Implement FedProx for PyTorch version
        # if cfg.fedprox:

        super(ClientModel,self).__init__(seed,lr)

    def create_model(self):
        net=build_net(self.dataset,self.model_name,self.num_classes)
        loss=nn.CrossEntropyLoss(reduction='none')
        optimizer=torch.optim.SGD
        optimizer_args={'lr':self.lr}
        return net,loss,optimizer, optimizer_args

    def test(self, data):
        self.net.eval()
        self.net = self.net.to(self.DEVICE)

        x_vecs = data['x'].to(self.DEVICE).float()
        labels = data['y'].to(self.DEVICE).long()

        output = self.net(x_vecs)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()
        # print("HAHA")
        # print(self.losses(output, labels), self.losses(output, labels).mean(), torch.mean(self.losses(output, labels)))
        loss = self.losses(output, labels).mean().item()
        self.net = self.net.to("cpu")

        return {ACCURACY_KEY: correct/len(data['x']), "loss": loss}

    def process_x(self, raw_x_batch):
        return torch.from_numpy(raw_x_batch)

    def process_y(self, raw_y_batch):
        return torch.from_numpy(raw_y_batch)