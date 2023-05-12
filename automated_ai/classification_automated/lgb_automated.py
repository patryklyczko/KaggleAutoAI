from ...classification.models.light_lgb import LightLGB
from ...model_data.model_data import ModelData
from ..algorithms.complexity_level0 import Complexity_Level0
from ..algorithms.complexity_level1 import Complexity_Level1
import logging
logging.basicConfig(level=logging.WARNING)

class LGBAutomated(ModelData):
    def __init__(self, cat):
        self.model = None
        self.columns_used = None
        
    def create(self, X, y, complexity):
        if complexity == 0:
            l = LightLGB()
            l.create_optuna(X,y,n_trials=200)
            self.model = l.get()
            self.columns_used = X.columns
        if complexity == 1:
            l = LightLGB()
            self.model, self.columns_used = Complexity_Level0(l, X, y)

    def get(self):
        return self.model
    
    def columns(self):
        return self.columns_used