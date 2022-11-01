import torch
import numpy as np
import torch.nn as nn
from torch import optim
import pickle
from torch.utils.data import DataLoader
from itertools import product

import models
from pdb_parser import ELEMENTS
from pdb_dataset import PDBDataset, my_collate


FILE_PATH = '/mnt/c/Users/zlils/Documents/university/biology/rhodopsins/excel/data.xlsx'


def main():
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.ConvModel(len(ELEMENTS), 10, 5)
    loss_module = nn.BCEWithLogitsLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=0.001)
    dataset = PDBDataset('./pdbs', FILE_PATH)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=my_collate)
    for batch in dataloader:
        for protein in batch[0]:
            model(protein)
        continue

# def save(self):
#     with open('./rhodopsin_model', 'wb') as f:
#         pickle.dump(self.model, f)
#
# def load(self, path: str, mode: str, config_path=None):
#     if config_path is not None:
#         # need to recreate the model according to the loaded dictionary
#         with open(config_path, 'rb') as f:
#             self.config = pickle.load(f)
#             self.__dict__.update(GraphGNN(self.config).__dict__)  # change self to have parameters of new model instead of what was in default config
#     checkpoint = torch.load(path, map_location=self.device)
#     self.model.load_state_dict(checkpoint['model_state_dict'])
#     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     self.loss_module = checkpoint['loss']
#     if not mode == 'eval':
#         # only load metrics if we continue training for some reason.
#         self.metrics_train = checkpoint['metrics_train']
#         self.metrics_val = checkpoint['metrics_val']
#
#     if mode == "eval":
#         print("model loaded, setting to eval mode")
#         self.model.eval()
#     else:
#         print("model loaded, setting to train mode")
#         self.model.train()
#
#
# def single_epoch(self, training_loader, epoch, single_node):
#     # start metrics
#     metrics_dict = dict()
#     avg_loss = 0
#     self.metrics_train.reset()
#     pbar = tqdm(iterable=training_loader, mininterval=30)
#     for data in pbar:
#         data.to(self.device)
#         self.optimizer.zero_grad()
#
#         x = self.model(data)
#         x = x.squeeze(dim=-1)
#         if single_node:
#             x = x[np.unique(data.batch.cpu(), return_index=True)[1]]
#
#         if self.config['model_dict']["c_out"] == 1:
#             predictions = (x > 0).float()
#             data.y = data.y.float()
#         else:
#             predictions = x.argmax(dim=-1)
#
#
#         loss = self.loss_module(x, data.y)
#         loss.backward()
#         avg_loss += loss.item()
#
#         self.optimizer.step()
#         batch_scores = self.metrics_train(predictions, data.y.int())
#         pbar.set_description(f"Train on Epoch {epoch}", refresh=False)
#         metrics_dict = {'loss': avg_loss / (epoch + 1)}
#         metrics_dict.update(batch_scores)
#         pbar.set_postfix(metrics_dict, refresh=False)
#     # self.scheduler.step()
#     metrics_dict.update(self.metrics_train.compute())
#     wandb.log(metrics_dict)
#     pbar.close()
#
# def train(self, training_loader, validation_loader, single_node=False):
#     self.model.to(self.device)
#     best_model_saver = SaveBestModel()
#     for epoch in range(self.config['train_dict']['epochs']):
#         # train
#         self.model.train()
#         self.single_epoch(training_loader, epoch, single_node)
#
#         # eval
#         self.model.eval()
#         if len(validation_loader):
#             self.val(validation_loader, epoch, single_node)
#             best_model_saver(self.metrics_val['AUROC'], self, epoch)
#         else:
#             best_model_saver(self.metrics_train['AUROC'], self, epoch)
#
# def val(self, validation_loader, epoch, single_node):
#     metrics_dict = dict()
#     valid_loss = 0
#     self.metrics_val.reset()
#     pbar = tqdm(iterable=validation_loader, mininterval=30)
#     # turn off gradients for validation
#     with torch.no_grad():
#         for data in pbar:
#             data.to(self.device)
#             # forward pass
#             x = self.model(data)
#             x = x.squeeze(dim=-1)
#             # validation batch loss
#             if single_node:
#                 x = x[np.unique(data.batch.cpu(), return_index=True)[1]]
#
#             if self.config['model_dict']["c_out"] == 1:
#                 predictions = (x > 0).float()
#                 data.y = data.y.float()
#             else:
#                 predictions = x.argmax(dim=-1)
#             loss = self.loss_module(x, data.y)
#             # accumulate the valid_loss
#             valid_loss += loss.item()
#             batch_scores = self.metrics_val(predictions, data.y.int())
#
#             metrics_dict = {'val_loss': valid_loss / (epoch + 1)}
#             metrics_dict.update(batch_scores)
#             pbar.set_description(f"Validation on Epoch {epoch}", refresh=False)
#             pbar.set_postfix(metrics_dict, refresh=False)
#     metrics_dict.update(self.metrics_val.compute())
#     print(self.metrics_val.compute())
#     wandb.log(metrics_dict)
#     pbar.close()
#
# def test(self, test_loader, thresholds, single_node=False, inference=False):
#     metrics_dict = dict()
#     self.model.eval()
#     test_loss = 0
#     self.metrics_val.reset()
#     roc = ROC()
#     pr = PrecisionRecallCurve()
#     pbar = tqdm(iterable=test_loader, mininterval=30)
#     test_thresh_auc = [AUROC().to(self.device) for _ in range(len(thresholds))]
#     ret_predictions = []
#     ret_labels = []
#     # turn off gradients for validation
#     with torch.no_grad():
#         for data in pbar:
#             data.to(self.device)
#             # forward pass
#             x = self.model(data)
#             x = x.squeeze(dim=-1)
#             # test batch loss
#             ret_predictions += torch.nn.Sigmoid()(x).tolist()
#             ret_labels += data.y.int().tolist()
#             if single_node:
#                 x = x[np.unique(data.batch.cpu(), return_index=True)[1]]
#
#             roc(torch.nn.Sigmoid()(x), data.y.int())
#             pr(torch.nn.Sigmoid()(x), data.y.int())
#             if self.config['model_dict']["c_out"] == 1:
#                 predictions = torch.nn.Sigmoid()(x).float()
#                 data.y = data.y.float()
#                 data.y_mg = {thresh: (torch.tensor(np.array(data.mg_dist)) < thresh).int().to(self.device) for thresh in thresholds}
#             else:
#                 predictions = x.argmax(dim=-1)
#             loss = self.loss_module(x, data.y)
#             # accumulate the valid_loss
#             test_loss += loss.item()
#             batch_scores = self.metrics_val(predictions, data.y.int())
#             for thresh, auc in zip(thresholds, test_thresh_auc):
#                 auc(predictions, data.y_mg[thresh])
#             metrics_dict = {'test_loss': test_loss}
#             metrics_dict.update(batch_scores)
#             pbar.set_description(f"Test on test set", refresh=False)
#             pbar.set_postfix(metrics_dict, refresh=False)
#     metrics_dict.update(self.metrics_val.compute())
#     fpr, tpr, roc_thresholds = roc.compute()
#     precision, recall, pr_thresholds = pr.compute()
#     if inference:
#         return ret_predictions
#     plot_curves(fpr, tpr, roc_thresholds, title="ROC", x_label="fpr", y_label='tpr')
#     plot_curves(precision, recall, pr_thresholds, title="Precision Recall Curve", x_label="precision", y_label='recall')
#     print(metrics_dict)
#     auc_results = []
#
#     for thresh, auc in zip(thresholds, test_thresh_auc):
#         result = auc.compute()
#         auc_results.append(result)
#         print(f"test auc with threshold of {thresh} is {result}")
#     data = [[x, y.item()] for (x, y) in zip(thresholds, auc_results)]
#     if not inference and wandb.run is not None:
#         table = wandb.Table(data=data, columns=["thresholds", "auc result"])
#         wandb.log({"auc to thresholds" : wandb.plot.line(table, "thresholds", "auc result",
#                                                          title="auc to thresholds")})
#     pbar.close()
#     return ret_predictions, ret_labels
#

if __name__ == "__main__":
    main()