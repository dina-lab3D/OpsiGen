import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, Recall, Precision, Specificity, AUROC
import torch.optim as optim
import numpy as np
import pickle


import configurations.consts as consts
from data_fetchers.pdb_dataset import PDBDataset
import training.models as models
from torch.utils.data import DataLoader

"""
Class responsible for creating and training Graph Level classification 
This class initializes the model, has a training and test function for it and is also responsible for loading and
saving models 
"""


class GraphGNN:

    def __init__(self, config=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = models.GNNModel(c_in=20, c_hidden=15, c_out=10, layer_name="GraphConv")
        self.loss_module = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        # self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer)
        metrics = MetricCollection([Accuracy(), Precision(), Recall(), Specificity(), AUROC()]).to(self.device)
        self.metrics_train = metrics.clone(prefix='train_')
        self.metrics_val = metrics.clone(prefix='val_')

    def save(self):
        path = consts.MODEL_PATH
        config_path = consts.CONFIG_PATH
        torch.save({"model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": self.loss_module,
                    "metrics_train": self.metrics_train,
                    "metrics_val": self.metrics_val},
                   path)
        with open(config_path, 'wb') as f:
            pickle.dump(dict(self.config), f)

    def load(self, path: str, mode: str, config_path=None):
        if config_path is not None:
            # need to recreate the model according to the loaded dictionary
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
                self.__dict__.update(GraphGNN(self.config).__dict__)  # change self to have parameters of new model instead of what was in default config
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_module = checkpoint['loss']
        self.metrics_train = checkpoint['metrics_train']
        self.metrics_val = checkpoint['metrics_val']

        if mode == "eval":
            print("model loaded, setting to eval mode")
            self.model.eval()
        else:
            print("model loaded, setting to train mode")
            self.model.train()

    def single_epoch(self, training_loader, epoch, single_node):
        # start metrics
        metrics_dict = dict()
        avg_loss = 0
        self.metrics_train.reset()
        pbar = tqdm(iterable=training_loader, mininterval=30)
        for data in pbar:
            # data.to(self.device)
            self.optimizer.zero_grad()

            x = self.model(data[0][0].float(), data[1][0])
            x = x.squeeze(dim=-1)
            if single_node:
                x = x[np.unique(data.batch.cpu(), return_index=True)[1]]

            # if False:
            #     predictions = (x > 0).float()
            #     data.y = data.y.float()
            # else:
            #     predictions = x.argmax(dim=-1)

            loss = self.loss_module(x, x)
            loss.backward()
            avg_loss += loss.item()

            self.optimizer.step()
            # batch_scores = self.metrics_train(predictions, data.y.int())
            pbar.set_description(f"Train on Epoch {epoch}", refresh=False)
            metrics_dict = {'loss': avg_loss / (epoch + 1)}
            # metrics_dict.update(batch_scores)
            pbar.set_postfix(metrics_dict, refresh=False)

            print("I am here!")

        # self.scheduler.step()
        metrics_dict.update(self.metrics_train.compute())
        wandb.log(metrics_dict)
        pbar.close()

    def train(self, training_loader, single_node=False):
        self.model.to(self.device)
        for epoch in range(40):
            # train
            self.model.train()
            self.single_epoch(training_loader, epoch, single_node)

            # eval
            self.model.eval()
            self.save()


def main():
    my_model = GraphGNN()
    train_dataloader = DataLoader(PDBDataset(consts.PDB_LIST, consts.PDBS_PATH), shuffle=True)
    my_model.train(training_loader=train_dataloader)


if __name__ == "__main__":
    main()
