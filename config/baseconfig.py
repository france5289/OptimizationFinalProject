import abc
class BaseConfig():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        ''' Should implement constructor here '''
    
    @abc.abstractmethod
    def load_from_json(self):
        ''' Should implement how to read hyperparameters from json files'''