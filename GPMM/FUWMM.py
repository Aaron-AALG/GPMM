import pandas as pd
import numpy as np
import scipy.optimize as opt

def FUWMM_requirements(data, L, U, norm, w0, p, defuzzification, display):
    '''
    Fuzzy-Unweighted-MM requirements
    ================================

    It checks whether the basic properties of FUWMM are satisfied.
    '''
    # data requirements
    if len(data.shape) < 2:
        raise ValueError('[!] data must be matrix-shaped.')
    elif data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError('[!] data must be matrix-shaped.')

    # Fuzzy-L requirements
    if len(L) != data.shape[1]:
        raise ValueError('[!] Number of lower bounds must be equal to the number of criteria.')
    if any([l < 0 or l > 1 for l in L]):
        raise ValueError('[!] Lower bounds must belong to [0,1].')
    if np.sum(L[np.arange(4) + 4]) > 1:
        raise ValueError('[!] The sum of the right spread of lower bounds must be less than 1.')

    # Fuzzy-U requirements
    if len(U) != data.shape[1]:
        raise ValueError('[!] Number of upper bounds must be equal to the number of criteria.')
    if any([u < 0 or u > 1 for u in U]):
        raise ValueError('[!] Upper bounds must belong to [0,1].')
    if np.sum(U[np.arange(4) + 4]) < 1:
        raise ValueError('[!] The sum of the right spread of upper bounds must be greater than 1.')

    # norm requirements
    if norm not in ["euclidean", "linear", "gauss", "max", "minmax", "standard", "none"]:
        raise ValueError('[!] The normalization method must be either "euclidean", "linear", "gauss", "max", "minmax", "standard" or "none".')

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

    # defuzzification requirements
    if defuzzification not in ['distance', 'max', 'min', 'sum', 'prod', 'center']:
        raise ValueError('[!] The defuzzification method must be either "min", "max", "sum", "prod", or "area".')

    # display requirements
    if int(display) not in [0,1]:
        raise ValueError('[!] "display" must be boolean.')

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

def from_fuzzy_to_score(fuzzy_number, defuzzification):
    '''
    Transformation from LR-fuzzy trapezoid to numeric score.
    =======================================================
    '''
    if defuzzification == 'distance':
        score = np.sum((fuzzy_number - np.zeros(4))**2)**(1/2)
    elif defuzzification == 'max':
        score = np.max(fuzzy_number)
    elif defuzzification == 'min':
        score = np.min(fuzzy_number)
    elif defuzzification == 'sum':
        score = np.sum(fuzzy_number)
    elif defuzzification == 'prod':
        score = np.prod(fuzzy_number)
    elif defuzzification == 'center':
        score = 1/2 * (fuzzy_number[0] + fuzzy_number[-1])
    return score

def fuzzy_score_i(w, data_norm, p, defuzzification, J, i, optimal_mode):
    '''
    Fuzzy-Score of the i-th alternative
    ===================================
    '''
    N_criteria = len(w) // 4
    alternative_i = np.array(data_norm)[i].reshape(N_criteria, 4)
    fuzzy_weights = w.reshape(N_criteria, 4)
    # First: Set the initial value
    for j in range(4):
        if p == 0:
            fuzzy_i = np.prod(alternative_i ** fuzzy_weights, axis = 0)
        elif p == 'min':
            fuzzy_i = alternative_i.min(axis = 0)
        elif p == 'max':
            fuzzy_i = alternative_i.max(axis = 0)
        else:
            fuzzy_i = np.sum((fuzzy_weights * alternative_i**p)**(1/p), axis = 0)
    score_i = from_fuzzy_to_score(fuzzy_i, defuzzification)

    # Decide whether we minimize or maximize the score
    if optimal_mode == 'min':
        score_i = score_i
    else:
        score_i = - score_i
    return score_i

def boolean_constraint(weights):
    '''
    Constraint function for the optimization of the FUWMM score
    ===========================================================

    It checks whether the fuzzy-weights have LR-fuzzy trapezoidal shape\n
    i.e. w_1 <= w_2 <= w_3 = w_4
    '''
    N_criteria = len(weights) // 4
    fuzzy_weights = weights.reshape(N_criteria, 4)
    LR_fuzzy_restriction = all([f_w[0] <= f_w[1] and f_w[1] <= f_w[2] and f_w[2] <= f_w[3] for f_w in fuzzy_weights])
    if LR_fuzzy_restriction:
        constraint = 1
    else:
        constraint = -1
    return constraint

def optimize_fuzzy_score(data_norm, L, U, w0, p, defuzzification, optimal_mode, display, Nrows, Ncols):
    '''
    Optimization stage for the fuzzy score
    ======================================
    '''
    # Bounds and constraints
    bounds = [(l, u) for l, u in zip(L, U)]
    constraints = ({'type': 'ineq', 'fun': lambda w: boolean_constraint(w)})

    # Optimizing the R-score according to the optimal_mode
    A = []
    w = []
    for i in range(Nrows):
        opt_i = opt.minimize(fun = fuzzy_score_i,
                             x0 = w0,
                             args = (data_norm, p, defuzzification, Ncols, i, optimal_mode),
                             method = 'SLSQP',
                             bounds = bounds,
                             constraints =  constraints,
                             tol = 10**(-16),
                             options = {'disp': display})
        if optimal_mode == 'max':
            opt_i.fun = - opt_i.fun
        A.append(opt_i.fun)
        w.append(opt_i.x)
    return A, w

def fuzzy_UWMM(data, L, U, norm = "none", w0 = [], p = 1, defuzzification = 'distance', display = False):
    """
    Fuzzy-Unweighted Mean Model
    ===========================

    Input:
    -------
        data: dataframe which contains the alternatives and the criteria as trapezoidal LR-fuzzy numbers (x1, x2, x3, x4), so that x1 <= x2 <= x3 <= x4.
        norm: normalization method for the data, whether "euclidean", "linear", "gauss", "max", "minmax", "standard" or "none".
        w0: array with the initial guess of the weights as trapezoidal LR-fuzzy numbers (w1, w2, w3, w4).
        defuzzification: Transformation from LR-fuzzy to score, whether "sum", "max", "min", "prod", "area".
        p: exponent of the weighted power mean, either "max", "min" or a float.
        display: logical argument to indicate whether to show print convergence messages or not.

    Output:
    -------
        Dictionary which contains three keys.
            Ranking: List with A_min and A_max fuzzy scores in regard of the optimal weights.
            Weights_min: List with the weights that minimizes the A score.
            Weights_max: List with the weights that maximizes the A score.
    """
    # Check whether the data input verifies the basic requirements
    FUWMM_requirements(data, L, U, norm, w0, p, defuzzification, display)

    # 1st: Normalize data
    data = np.array(data)
    Nrows, Ncols = data.shape
    data_norm = fuzzy_normalization(data, norm, Ncols)

    # 2nd: Set initial weights
    if w0 == []:
        w0 = [1/Ncols for _ in range(Ncols)]

    # 3rd: Fuzzy computation
    fuzzy_min, w_min = optimize_fuzzy_score(data_norm, L, U, w0, p, defuzzification, 'min', display, Nrows, Ncols)
    fuzzy_max, w_max = optimize_fuzzy_score(data_norm, L, U, w0, p, defuzzification, 'max', display, Nrows, Ncols)

    # 4th Prepare ranking system
    fuzzy_scores = {'fuzzy_min': fuzzy_min, 'fuzzy_max': fuzzy_max}
    output_fuzzy_uwSM = {'Ranking': fuzzy_scores, 'Weights_min': w_min, 'Weights_max': w_max}
    return output_fuzzy_uwSM
