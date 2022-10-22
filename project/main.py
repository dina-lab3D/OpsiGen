import train
from train_utils import TrainerData

if __name__ == "__main__":
    td = TrainerData("./data.txt")
    trainer = train.Trainer(td)
    trainer.train()
