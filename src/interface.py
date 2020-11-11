from abc import ABC, abstractmethod


class EmbeddingInterface(ABC):
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
