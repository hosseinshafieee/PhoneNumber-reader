import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = "Models/"
        self.vocab = "0123456789"
        self.height = 32
        self.width = 128
        self.max_text_length = 11
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.train_epochs = 500
        self.train_workers = 20