import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import NCF
from .metric import accuracy

class Trainer:
    def __init__(self, model: NCF, data_loader: DataLoader, lr: float = 1e-4,
                 with_cuda: bool = True, cuda_device=True, cuda_devices=None):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.model = model

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.data_loader = data_loader
        self.optim = Adam(self.model.parameters(), lr=lr)

        self.criterion = nn.BCELoss()

        print("Total Model Parameter:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        log_dict = {"epoch": [], "iteration": [], "avg_item_loss": [],
                    "avg_user_loss": [], "avg_domain_loss": [], "avg_top_k": []}
        data_iter = tqdm.tqdm(enumerate(self.data_loader),
                              desc="EP_train:%d" % epoch,
                              total=len(self.data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        avg_acc = 0.0

        for i, data in data_iter:
            output = self.model(data["user"], data["item"])

            loss = self.criterion(output.to(torch.float), data["label"].to(torch.float))
            acc = accuracy(output, data["label"])

            loss.backward()
            self.optim.step()

            avg_loss += loss.item()
            avg_acc += acc.item()

            post_fix = {
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": avg_acc / (i + 1)
            }

            data_iter.set_postfix(post_fix)

        print(f"EP{epoch}_train, avg_loss={avg_loss}, avg_acc={avg_acc}")


    def save(self, epoch, file_path="output", model_name="model"):
        output_path = file_path + '/' + model_name + "ep.%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print(f"EP:{epoch} Model saved on {output_path}")

        return output_path

