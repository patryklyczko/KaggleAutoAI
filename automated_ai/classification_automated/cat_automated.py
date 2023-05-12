from ...classification.models.cat import Cat
from ...model_data.model_data import ModelData
from ..algorithms.complexity_level0 import Complexity_Level0
from ..algorithms.complexity_level1 import Complexity_Level1
import logging
logging.basicConfig(level=logging.WARNING)

class CatAutomated(ModelData):
    def __init__(self, cat):
        self.model = None
        self.columns_used = None
        
    def create(self, X, y, complexity):
        if complexity == 0:
            cat = Cat()
            cat.create_optuna(X,y,n_trials=200)
            self.model = cat.get()
            self.columns_used = X.columns
        if complexity == 1:
            cat = Cat()
            self.model, self.columns_used = Complexity_Level0(cat, X, y)

    def get(self):
        return self.model
    
    def columns(self):
        return self.columns_used