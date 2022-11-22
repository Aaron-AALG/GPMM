Generalized *p*-Mean Models
================================

![python-version](https://img.shields.io/badge/python->=3.8-orange.svg)
[![pypi-version](https://img.shields.io/pypi/v/gpmm.svg)](https://pypi.python.org/pypi/gpmm/)
![license](https://img.shields.io/pypi/l/gpmm.svg)
[![Downloads](https://static.pepy.tech/personalized-badge/gpmm?period=total&units=none&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/gpmm)


Collection of Generalized *p*-Mean Models (GPMM) with classic, fuzzy and un-weighted approach. This set of outranking methods are based on the concept of [weighted generalized p-mean](https://en.wikipedia.org/wiki/Generalized_mean) of a sequence:

$$ M_i^p(X,w) = \left[\sum_{j=1}^M w_jx_{ij}^p\right]^\frac{1}{p} $$

In this project, we have included four different approaches:

1. **Classic (WMM)**: The *M* score is computed per each alternative to generate a cardinal ranking.
2. **Fuzzy (FWMM)**: The decision matrix is trapezoidal fuzzy shaped as $(x_L, x_1, x_2, x_R)$ with LR-representation. Then, it is satisfied that $x_L \le x_1 \le x_2 \le x_R$ per each component of the matrix.
3. **Unweighted (UWMM)**: The weighting scheme is variable and it has attached a lower and upper bound per each component. As a result, it returns an interval $[M_L, M_U]$.
4. **Fuzzy Un-Weighted (FUWMM)**: It combines both approaches in the decision matrix and the weighting scheme.

The mathematical fuzzy LR-representation of a trapezoid $(x_L, x_1, x_2, x_R)$ is depicted as follows:

![x_fuzzy](images/x_fuzzy.png)


Installation
--------------------------------

You can install the GPMM library from GitHub:

```terminal
git clone https://github.com/Aaron-AALG/GPMM.git
python3 -m pip install -e GPMM
```

You can also install it directly from PyPI:

```terminal
pip install GPMM
```

Example
--------------------------------

GPMM is implemented in order to manage **NumPy** arrays. Here is an example in which we only use three alternatives and four criteria.

```python
import pandas as pd
import numpy as np
from GPMM.UWMM import UWMM

data = pd.DataFrame({"c1":[173, 176, 142],
                    "c2":[10, 11, 5],
                    "c3":[11.4, 12.3, 8.2],
                    "c4":[10.01, 10.48, 7.3]})
directions = ["max", "max", "min", "min"]
L = np.repeat(0.1, data.shape[1])
U = np.repeat(0.4, data.shape[1])
p = 2

x = UWMM(data, directions, L, U, p=p)
```

Optimization in Python
--------------------------------

This library uses the [minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) function of the scipy.optimize module to carry out the optimization problems. In particular, $M_L$ and $M_U$ are obtained one by one, thus we can apply the **SLSQP** method.
