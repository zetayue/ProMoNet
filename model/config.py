class Config(object):
    """
    Model configurations
    """
    def __init__(self, dim_interact, dim_pair, n_module, dropout=0, factor=0, task='NULL'):
        self.dim_interact = dim_interact
        self.dim_pair = dim_pair
        self.n_module = n_module
        self.dropout = dropout
        self.factor = factor
        self.task = task