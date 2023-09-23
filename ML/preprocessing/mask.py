import numpy as np
def mask_preprocess(X, size=200):
    return 255-X[:size,:]
