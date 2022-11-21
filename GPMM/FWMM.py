import numpy as np

def FWMM_requirements(data, w, p, norm, defuzzification):
    # data requirements
    if len(data.shape) < 2:
        raise ValueError('[!] data must be matrix-shaped.')
    elif data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError('[!] data must be matrix-shaped.')
    elif data.shape[1]%4 != 0:
        raise ValueError('[!] data arguments must be trapezoid-fuzzy-shaped: [L, x1, x2, R] where L <= x1 <= x2 <= R.')
    
    # w requirements
    if len(w) > 0:
        if len(w) != data.shape[1]:
            raise ValueError('[!] Length of initial weight must be equal to the number of criteria.')
        if all([w < 0 for w in w]) and all([w > 1 for w in w]):
            raise ValueError('[!] Initial weights must belong to [0,1].')
    
    # p requirements
    if p not in ["min", "max"] and not any([isinstance(p,int), isinstance(p,float)]):
            raise ValueError('[!] The p value must be either "min", "max", or a float number.') 
    
    # norm requirements
    if norm not in ["euclidean", "linear", "gauss", "max", "minmax", "standard", "none"]:
        raise ValueError('[!] The normalization method must be either "euclidean", "linear", "gauss", "max", "minmax", "standard" or "none".')

    # defuzzification requirements
    if defuzzification not in ['distance', 'max', 'min', 'sum', 'prod', 'center']:
        raise ValueError('[!] The defuzzification method must be either "min", "max", "sum", "prod", or "area".')
    
    return

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

def fuzzy_score(w, data_norm, p):
    '''
    #  Computes the fuzzy generalized p-mean per each alternative.
    In order to compute fuzzy operators, we split the data in the (x_L, x_1, x_2, x_R) column-components.
    '''
    # Extraction of indexes of the fuzzy-components
    fuzzy_index = [np.arange(data_norm.shape[1])%4 == i for i in range(4)]
    
    # Create the fuzzy-score with (x_L, x_1, x_2, x_R)-shape
    f_score = np.zeros((data_norm.shape[0], 4))
    # Compute the A-score per each fuzzy component
    for i, idx in enumerate(fuzzy_index):
        if p == 0:
            A_fuzzy = np.prod(data_norm[:,idx]**w[idx], axis=1)
        elif p == 'min':
            A_fuzzy = np.min(data_norm[:,idx]**p, axis=1)
        elif p == 'max':
            A_fuzzy = np.max(data_norm[:,idx]**p, axis=1)
        else:
            A_fuzzy = np.sum(data_norm[:,idx]**p * w[idx], axis=1)**(1/p)
        f_score[:,i] = A_fuzzy
    return f_score

def from_fuzzy_to_score(fuzzy_number, defuzzification):
    '''
    Transformation from LR-fuzzy trapezoid to numeric score.
    =======================================================
    '''
    if defuzzification == 'distance':
        score = np.sum((fuzzy_number - np.zeros(4))**2, axis = 1)**(1/2)
    elif defuzzification == 'max':
        score = np.max(fuzzy_number, axis = 1)
    elif defuzzification == 'min':
        score = np.min(fuzzy_number, axis = 1)
    elif defuzzification == 'sum':
        score = np.sum(fuzzy_number, axis = 1)
    elif defuzzification == 'prod':
        score = np.prod(fuzzy_number, axis = 1)
    elif defuzzification == 'center':
        score = 1/2 * (fuzzy_number[0] + fuzzy_number[-1])
    return score

def fuzzyWMM(data, w, p = 1, norm = "none", defuzzification = 'distance'):
    """
    Fuzzy-Weighted Mean Models
    ==========================

    Input:
    -------
        data: dataframe which contains the alternatives and the criteria as fuzzy numbers (x_L, x_1, x_2, x_R).
        w: array with the weights as fuzzy numbers (w_L, x_1, x_2, w_R).
        p: exponent of the weighted power mean.
        norm: normalization method for the data, whether "euclidean", "linear", "gauss", "max", "minmax", "standard" or "none".
        comparison: fuzzy
    
    Output:
    -------
        Dictionary which contains three keys.
            fuzzy_data: Array with the alternatives and their fuzzy score.
            Ranking: Array with fuzzyWMM scores.
    """
    # Check whether the data input verifies the basic requirements
    FWMM_requirements(data, w, p, norm, defuzzification)
    
    # 1st: Normalize data
    data = np.array(data)
    Nrow, Ncol = data.shape
    data_norm = fuzzy_normalization(data, norm, Ncol)

    # 2nd: Fuzzy computation
    fuzzy_data = fuzzy_score(w, data_norm, p)
    ranking = from_fuzzy_to_score(fuzzy_data, defuzzification)
    FWMM_result = {'fuzzy_output': fuzzy_data, 'ranking': ranking}

    return FWMM_result

