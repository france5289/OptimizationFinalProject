import json 
import os

from config.baseconfig import BaseConfig

class AdamConfig(BaseConfig):
    '''
    Use this object to load  config ResNet-18 model with Adam Optimizer from json files
    '''
    def __init__(self):
        self.expname = 'Default'
        self.seed = 1024
        self.lrate = 1e-3
        self.nepoch = 10
        self.weight_decay = 0
        
    # ==== Getter ====
    @property
    def expname(self):
        return self._expname
    @property
    def seed(self):
        return self._seed 
    @property
    def lrate(self):
        return self._lrate
    @property
    def nepoch(self):
        return self._nepoch
    @property
    def weight_decay(self):
        return self._weight_decay
    # ==== Setter ====
    @expname.setter
    def expname(self, value):
        if type(value) is not str:
            raise ValueError('expname should be a string!')
        self._expname = value
    @seed.setter
    def seed(self, value):
        if type(value) is not int:
            raise ValueError('Random seed should be an integer')
        self._seed = value
    @lrate.setter
    def lrate(self, value):
        if type(value) is not float:
            raise ValueError('learning rate should be a float')
        self._lrate = value
    @nepoch.setter
    def nepoch(self, value):
        if type(value) is not int:
            raise ValueError('Num of epoch should be an integer')
        self._nepoch = value
    @weight_decay.setter
    def weight_decay(self, value):
        self._weight_decay = value
    
    @classmethod
    def load_from_json(cls, jsonpath):
        '''
        Factory method to read hyperparameter settings from json files
        and return a new class
        
        Args:
        --------
            jsonpath(string): json file path
        
        Return:
        --------
            self(Config obeject)
        '''
        self = cls()

        if jsonpath is None or type(jsonpath) != str:
            raise ValueError('Arg `jsonpath` should be a string!')
        if not os.path.exists(jsonpath):
            raise FileNotFoundError(f'File {jsonpath} does not exits')
        print('Reading Hyperparameter setting')
        with open(jsonpath, 'r') as f:
            hyper_params = f.read()
        obj = json.loads(hyper_params)
        # ============ set hyperparameters ===============
        self.expname = obj['expname']
        self.seed = obj['seed']
        self.lrate = obj['lrate']
        self.weight_decay = obj['weight_decay']
        print('All hyperparameters have been read in')
        return self