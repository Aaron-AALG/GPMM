import numpy as np
import scipy.optimize as opt

def UWMM_requirements(data, L, U, norm, w0, p, display):
    
    # data requirements
    if len(data.shape) < 2:
        raise ValueError('[!] data must be matrix-shaped.')
    elif data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError('[!] data must be matrix-shaped.')
    
    # norm requirements
    if norm not in ["euclidean", "linear", "gauss", "max", "minmax", "standard", "none"]:
        raise ValueError('[!] The normalization method must be either "euclidean", "linear", "gauss", "max", "minmax", "standard" or "none".')

    # L requirements
    if len(L) != data.shape[1]:
        raise ValueError('[!] Number of lower bounds must be equal to the number of criteria.')
    if any([l < 0 or l > 1 for l in L]):
        raise ValueError('[!] Lower bounds must belong to [0,1].')
    if np.sum(L) > 1:
        raise ValueError('[!] The sum of lower bounds must be less than 1.')
    
    # U requirements
    if len(U) != data.shape[1]:
        raise ValueError('[!] Number of upper bounds must be equal to the number of criteria.')
    if any([u < 0 or u > 1 for u in U]):
        raise ValueError('[!] Upper bounds must belong to [0,1].')
    if np.sum(U) < 1:
        raise ValueError('[!] The sum of upper bounds must be greater than 1.')
    
    # w0 requirements
    if len(w0) > 0:
        if len(w0) != data.shape[1]:
            raise ValueError('[!] Length of initial weight must be equal to the number of criteria.')
        if all([w < 0 for w in w0]) and all([w > 1 for w in w0]):
            raise ValueError('[!] Initial weights must belong to [0,1].')
        if abs(np.sum(w0) - 1) > 10**(-6):
            raise ValueError('[!] Initial weights must sum 1.')
    
    # p requirements
    if p not in ["min", "max"] and not any([isinstance(p,int), isinstance(p,float)]):
            raise ValueError('[!] The p value must be either "min", "max", or a float number.')

    # display requirements
    if int(display) not in [0,1]:
        raise ValueError('[!] "display" must be boolean.')
    
    return

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

def WMM_score(w, data_norm, p):
    if p == 0:
        wsm = np.prod(data_norm**w, axis=1)
    elif p == 'min':
        wsm = np.min(w*data_norm**p, axis=1)
    elif p == 'max':
        wsm = np.max(w*data_norm**p, axis=1)
    else:
        wsm = np.sum(w*data_norm**p, axis=1)**(1/p)
    return wsm

def WMM_i(w0, data_norm, p, i, optimal_mode):
    wsm = WMM_score(w0, data_norm, p)
    if optimal_mode == 'min':
        wmm_i = wsm[i]
    else:
        wmm_i = -wsm[i]
    return wmm_i

def optimize_WMM(data_norm, w0, L, U, p, optimal_mode, display, N_alternatives):
    
    bounds = [(l,u) for l, u in zip(L, U)]
    constraints = ({'type': 'ineq', 'fun': lambda w: 1-np.sum(w)},
                   {'type': 'ineq', 'fun': lambda w: np.sum(w)-1},)
    
    # Optimizing the WSM-score according to the optimal_mode
    WMM = []
    weights = []
    for i in range(N_alternatives):
        # For Bounded-Constrained problems, we may apply either L-BFGS_B, Powell or TNC methods
        opt_i = opt.minimize(fun = WMM_i,
                            x0 = w0,
                            args = (data_norm, p, i, optimal_mode),
                            method = 'SLSQP',
                            bounds = bounds,
                            constraints =  constraints,
                            tol = 10**(-16),
                            options = {'disp': display})
        if optimal_mode == 'max':
            opt_i.fun = -opt_i.fun
        WMM.append(opt_i.fun)
        weights.append(opt_i.x)
    return WMM, weights

def UWMM(data, L, U, norm = "none", w0=[], p=1, display = False):
    """
    Unweighted Mean Models
    ======================
    UnWeighted Mean Models (UWMM) is a family of unweighted outranking MCDM methods 
    whose score function is the p-weighted norm of each alternative. As a consequence,
    no weighting scheme is requeried. Instead, UWMM makes use of lower and upper bounds
    to optimize the WMM score and then return the optimal values and solutions.

    Input:
    -------
        data: dataframe which contains the alternatives and the criteria.
        L: array with the lower bounds of the weigths.
        U: array with the upper bounds of the weigths.
        norm: normalization method for the data, whether "euclidean", "linear", "gauss", "max", "minmax", "standard" or "none".
        w0: array with the initial guess of the weights.
        p: exponent of the weighted power mean.
        display: logical argument to indicate whether to show print convergence messages or not.
    
    Output:
    -------
        Dictionary which contains three keys.
            Ranking: List with A_min and A_max scores in regard of the optimal weights.
            Weights_min: List with the weights that minimizes the A score.
            Weights_max: List with the weights that maximizes the A score.
    """
    # Check whether the data input verifies the basic requirements
    UWMM_requirements(data, L, U, norm, w0, p, display)
    
    # 1st step: Normalize data
    data = np.array(data)
    Nrows, Ncols = data.shape
    data_norm = normalization(data, norm)

    # 2nd: Set initial weights
    if w0 == []:
        w0 = [1/Ncols for _ in range(Ncols)]
    
    # 3rd step: Optimize R score
    r_min, w_min = optimize_WMM(data_norm, w0, L, U, p, 'min', display, Nrows)
    r_max, w_max = optimize_WMM(data_norm, w0, L, U, p, 'max', display, Nrows)
    
    # Output preparation
    scores = {'A_min': r_min, 'A_max': r_max}
    output_UWMM = {'Ranking': scores, 'Weights_min': w_min, 'Weights_max': w_max}

    return output_UWMM
