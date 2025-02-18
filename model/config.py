class Config(object):
    """
    Model configurations
    """
    def __init__(self, dim_interact, dim_pair, n_module, dropout):
        self.dim_interact = dim_interact
        self.dim_pair = dim_pair
        self.n_module = n_module
        self.dropout = dropout