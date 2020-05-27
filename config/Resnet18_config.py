import json 
import os

from config.baseconfig import BaseConfig

class AdamConfig(BaseConfig):
    '''
    Use this object to load config of Adam Optimizer from json files
    '''
    def __init__(self):
        '''
        Consturctor of AdamConfig object
        Note
        ---
            all hyperparameters' value will set to -1 or 0 
            please remeber to use load_from_json to load correct hyperparameters  
        '''
        self.expname = 'Default'
        self.seed = -1
        self.batch_size = -1.0
        self.lrate = -1.0
        self.nepoch = 0
        self.weight_decay = -1.0
        
        
    # ==== Getter ====
    @property
    def expname(self):
        return self._expname
    @property
    def seed(self):
        return self._seed 
    @property
    def batch_size(self):
        return self._batch_size
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
    @batch_size.setter
    def batch_size(self, value):
        if type(value) is not int:
            raise ValueError('Batch Size should be an integer')
        self._batch_size = value
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
        self.nepoch = obj['nepoch']
        self.batch_size = obj['batch_size']
        print('All hyperparameters have been read in')
        return self


class RangerConfig(BaseConfig):
    '''
    Use this object to load config of Ranger Optimizer from json files
    '''
    def __init__(self):
        '''
        Consturctor of AdamConfig object
        Note
        ---
            all hyperparameters' value will set to -1 or 0 
            please remeber to use load_from_json to load correct hyperparameters  
        '''
        self.expname = 'Default'
        self.seed = -1
        self.batch_size = -1.0
        self.lrate = -1.0
        self.nepoch = 0
        self.weight_decay = -1.0
        self.k = 0
        self.N_sma_threshold = 0
        self.eps = 0.0
        self.use_gc = True 
        self.gc_conv_only = False

    # ==== Getter ====
    @property
    def expname(self):
        return self._expname
    @property
    def seed(self):
        return self._seed 
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def lrate(self):
        return self._lrate
    @property
    def nepoch(self):
        return self._nepoch
    @property
    def weight_decay(self):
        return self._weight_decay
    @property
    def k(self):
        return self._k
    @property
    def N_sma_threshold(self):
        return self._N_sma_threshold
    @property
    def eps(self):
        return self._eps
    @property
    def use_gc(self):
        return self._use_gc
    @property
    def gc_conv_only(self):
        return self._gc_conv_only
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
    @batch_size.setter
    def batch_size(self, value):
        if type(value) is not int:
            raise ValueError('Batch Size should be an integer')
        self._batch_size = value
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
    @k.setter
    def k(self, value):
        if type(value) is not int:
            raise ValueError('k should be an integer')
        self._k = value
    @N_sma_threshold.setter
    def N_sma_threshold(self, value):
        if type(value) is not int:
            raise ValueError('N_sma_threshold should be an integer')
        self._N_sma_threshold = value
    @eps.setter
    def eps(self, value):
        if type(value) is not float:
            raise ValueError('eps should be a float')
        self._eps = value
    @use_gc.setter
    def use_gc(self, value):
        if type(value) is not bool:
            raise ValueError('use_gc should be a boolean')
        self._use_gc = value
    @gc_conv_only.setter
    def gc_conv_only(self, value):
        if type(value) is not bool:
            raise ValueError('gc_conv_only should be a boolean')
        self._gc_conv_only = value    


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
        self.nepoch = obj['nepoch']
        self.batch_size = obj['batch_size']
        self.k = obj['k']
        self.N_sma_threshold = obj['N_sma_threshold']
        self.eps = obj['eps']
        self.use_gc = obj['use_gc']
        self.gc_conv_only = obj['gc_conv_only']
        print('All hyperparameters have been read in')
        return self