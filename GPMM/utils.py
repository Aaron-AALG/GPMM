import numpy as np

#
# Normalization for STANDARD data
#
def normalization(data, norm):
    '''
    Normalization function for the decision matrix
    ==============================================

    Input:
    ------
        data: DataFrame with the decision matrix
        norm: Normalization type to apply over data

    Output:
    -------
        data_normalized: np.array with the normalized data
    '''
    # Determine the type of normalization
    if norm == 'euclidean':
        norm_func = lambda x: x / np.linalg.norm(x)
    elif norm == 'linear':
        norm_func = lambda x: x / np.sum(x)
    elif norm == 'minmax':
        norm_func = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    elif norm == 'max':
        norm_func = lambda x: x / np.max(x)
    elif norm == 'standard':
        norm_func = lambda x: (x - x.mean()) / np.std(x)
    elif norm == 'gauss':
        norm_func = lambda x: 1/(np.std(x) * np.sqrt(2*np.pi)) * np.exp(-1/2*((x-np.mean(x))/np.std(x))**2)
    elif norm == 'none':
        norm_func = lambda x: x

    # Apply normalization per each fuzzy-criteria
    data_normalized = np.zeros(data.shape)
    for criteria in range(data_normalized.shape[1]):
        data_normalized[:, criteria] = norm_func(data[:, criteria])

    return data_normalized

#
# Normalization for FUZZY data
#
def fuzzy_normalization(data, norm, Ncol):
    '''
    Normalization function for the fuzzy-data.
    ==========================================

    Input:
    ------
        data: DataFrame of the fuzzy decision matrix
        norm: Normalization type to apply over data

    Output:
    -------
        data_LR_fuzzy_normalized: np.array with the normalized data
    '''
    # Convert to LR-fuzzy
    data_LR_fuzzy = np.array([data[: , np.arange(4) + 4*i] for i in range(Ncol//4)])
    
    # Determine the type of normalization
    if norm == 'euclidean':
        norm_func = lambda x: x / np.linalg.norm(x)
    elif norm == 'linear':
        norm_func = lambda x: x / np.sum(x)
    elif norm == 'minmax':
        norm_func = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    elif norm == 'max':
        norm_func = lambda x: x / np.max(x)
    elif norm == 'standard':
        norm_func = lambda x: (x - x.mean()) / np.std(x)
    elif norm == 'gauss':
        norm_func = lambda x: 1/(np.std(x) * np.sqrt(2*np.pi)) * np.exp(-1/2*((x-np.mean(x))/np.std(x))**2)
    elif norm == 'none':
        norm_func = lambda x: x

    # Apply normalization per each fuzzy-criteria
    for i in range(data_LR_fuzzy.shape[0]):
        data_LR_fuzzy[i] = norm_func(data_LR_fuzzy[i])

    # Return as (I, J)-shape
    data_LR_fuzzy_normalized = np.concatenate(data_LR_fuzzy, 1)
    return data_LR_fuzzy_normalized
