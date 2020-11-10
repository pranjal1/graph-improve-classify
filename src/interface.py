from abc import ABC, abstractmethod


class DataloaderInterface(ABC):
    @abstractmethod
    def load_model(self):
        # if no model, this could be an identity function
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def get_embedding(self):
        pass

    @abstractmethod
    def train_test_dataloader(self):
        pass
