import os
import torch


class GetConfig:
    r"""
    Over all config for this repo.
    """
    def __init__(self, ratio=0.1, device="cuda:0", batch_size=64):
        # Base config
        self.ratio = ratio  # opt is the args parser in train_deep.py, ratio means scaling factor.
        self.channel = 1  # gray image.

        self.block_size = 256  # 256
        self.batch_size = 64
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Attention head
        self.head = 8

        # Paths of models
        self.results = "./results"
        self.log = os.path.join(self.results, str(int(self.ratio * 100)), "log.txt")

        self.folder = os.path.join(self.results, str(int(self.ratio * 100)), "models")
        self.model = os.path.join(self.folder, "model.pth")
        self.second = os.path.join(self.folder, "second.pth")
        self.optimizer = os.path.join(self.folder, "optimizer.pth")
        self.scheduler = os.path.join(self.folder, "scheduler.pth")
        self.info = os.path.join(self.folder, "info.pth")

        # Paths of datasets (will be packaged into *.pt)
        self.train_path = "./dataset/train"
        if not os.path.isdir(self.train_path):
            os.mkdir(self.train_path)
        self.val_path = "./dataset/val"
        if not os.path.isdir(self.val_path):
            os.mkdir(self.val_path)

        self.test_path = "./dataset/test"  # BSD58 & SET11

    def check(self):
        r"""
        After check func, your trained models dir will be like:
        ProjectName/result
                       |
                       |-- 10 (ratio * 100)
                       |    |
                       |    |-- init_model-- *.pth (best state_dict) & init_log.txt
                       |    |
                       |    |-- model-- *.pth
                       |    |
                       |    |-- log.txt
                       |
                       |-- 20
                       |    |
                       |    |-- init_model-- *.pth
                       |    |
                       |    |-- deep_model-- *.pth
                      ...
        """
        if not os.path.isdir(self.results):
            os.mkdir(self.results)

        sub_path = os.path.join(self.results, str(int(self.ratio * 100)))
        if not os.path.isdir(sub_path):
            os.mkdir(sub_path)
            print("Mkdir: " + sub_path)

        models_path = os.path.join(sub_path, "models")
        if not os.path.isdir(models_path):
            os.mkdir(models_path)
            print("Mkdir: " + models_path)

    def show(self):
        print("\n=> Your configs are:")
        print("=" * 70)
        for item in self.__dict__:
            print("{:<20s}".format(item + ":") + "{:<30s}".format(str(self.__dict__[item])))
            print("-" * 70)
        print("\n")


if __name__ == "__main__":
    config = GetConfig()
    config.check()
    config.show()
