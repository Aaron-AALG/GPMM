import numpy as np
from utils import normalization

def WMM_requirements(data, norm, w, p):
    
    # data requirements
    if len(data.shape) < 2:
        raise ValueError('[!] data must be matrix-shaped.')
    elif data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError('[!] data must be matrix-shaped.')
    
    # norm requirements
    if norm not in ["euclidean", "linear", "gauss", "max", "minmax", "standard", "none"]:
        raise ValueError('[!] The normalization method must be either "euclidean", "linear", "gauss", "max", "minmax", "standard" or "none".')
    
    # w requirements
    if len(w) > 0:
        if len(w) != data.shape[1]:
            raise ValueError('[!] Length of initial weight must be equal to the number of criteria.')
        if all([w < 0 for w in w]) and all([w > 1 for w in w]):
            raise ValueError('[!] Initial weights must belong to [0,1].')
        if abs(np.sum(w) - 1) > 10**(-6):
            raise ValueError('[!] Initial weights must sum 1.')
    
    # p requirements
    if p not in ["min", "max"] and not any([isinstance(p,int), isinstance(p,float)]):
            raise ValueError('[!] The p value must be either "min", "max", or a float number.')

    return

def WMM_score(w, data_norm, p):
    """
    WMM ranking score
    =================

    Input:
    ------
        w: np.array with the weights.
        data_norm: np.matrix with the normalized decision matrix.
        p: exponent of the weighted power mean.

    Output:
    -------
        Ranking score computed as the p-weighted power mean.
    """
    if p == 0:
        wsm = np.prod(data_norm ** w, axis=1)
    elif p == 'min':
        wsm = np.min(data_norm, axis=1)
    elif p == 'max':
        wsm = np.max(data_norm, axis=1)
    else:
        wsm = np.sum(w * data_norm**p, axis=1)**(1/p)
    return wsm

def WMM(data, w, norm = "none", p = 1):
    """
    WMM method
    ==========
    Weighted Mean Models (WMM) is a family of outranking MCDM methods 
    whose score function is the p-weighted norm of each alternative.

    Input:
    -------
        data: dataframe which contains the alternatives and the criteria.
        w: np.array with the weights.
        norm: normalization method for the data, whether "euclidean", "linear", "gauss", "max", "minmax", "standard" or "none".
        p: exponent of the weighted power mean.

    Output:
    -------
        WMM score.
    """
    # Check whether the data input verifies the basic requirements
    WMM_requirements(data, norm, w, p)
    
    # Normalize data
    data = np.array(data)
    data_norm = normalization(data, norm)

    return WMM_score(w, data_norm, p)
