from abc import ABC, abstractmethod
import numpy as np


class AbstractRegretMinimizer(ABC):
    def __call__(self, context):
        return self.nextAction(context)

    @abstractmethod
    def nextAction(self, context):
        raise NotImplementedError("subclass should implement this")

    @abstractmethod
    def observeLoss(self, loss):
        raise NotImplementedError("subclass should implement this")
    
    @abstractmethod
    def reset(self, loss):
        raise NotImplementedError("subclass should implement this") 