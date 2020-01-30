"""
Graphical Concord: sparse inverse covariance estimation with an L1 and Frobenius norm penalized estimator.
"""

# Author: Zhipu Zhou <zhipu@ucsb.edu>
# License: MIT
# Copyright: Sang-Yun Oh, Zhipu Zhou

import numpy as np
import numpy.ctypeslib as npct
import site
import os.path
import sys
from ctypes import c_int, c_double
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_array


# define new data types
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=c_int, ndim=1, flags='CONTIGUOUS')


# Helper functions that assist the implementation in the function graphical_concord
def _findpath():
    """Find the path to the shared library file: sharedlib.so
    
    The shared library file is the core file that implements the three solvers of 
    graphical concord and is produced from cpp files. Any changes of the solvers only
    need a change to the cpp files and the reproduction of the shared library.
    """
    paths = sys.path
    if len(paths) == 0:
        raise Exception('No path exists. Check with sys.path.')
    for path in paths:
        if os.path.isfile(path+"/gconcord/sharedlib.so"):
            return path+"/gconcord/sharedlib.so"
    raise Exception("No shared library in any paths. Please check the installation.")

    
def _createout(p):
    """Create the initial input omega for the iteration.
    
    The initial omega is set as an identity matrix, and only the information
    of non-zero elements is recorded to save memory: out for non-zero values, outi for row-index
    of non-zeros, and outj for column-index of nonzeros.
    """
    out = np.diag(np.repeat(1.0, p)).reshape(p**2,) + np.repeat(1.0, p**2)
    outi = np.array(np.repeat(-1, p**2), dtype = np.int32)
    outj = np.array(np.repeat(-1, p**2), dtype = np.int32)
    return out, outi, outj


def _createomega(out, outi, outj, p):
    """Recover the output omega from the information of non-zero elements.
    
    Recover the output omega from the value array out, the row index array outi, and 
    the column index array outj.
    """
    omega = np.zeros((p,p))
    for j in range(p**2):
        if outi[j] != -1:
            omega[outi[j], outj[j]] = out[j]
        else:
            break
    return omega
    
    
def _checkmethod(method):
    """Convert method of a string-type into a numerical type.
    """
    if method.lower() == "coordinatewise":
        return 1
    elif method.lower() == "ista":
        return 2
    elif method.lower() == "fista":
        return 3
    else:
        raise Exception('Incorrect method input.')
        
    
def _checklam(lam1, p):
    """Pre-conditions for the L1 penalty.
    
    If the input lam1 is a scalar, convert it into a matrix parameter with all 
    off-diagonal elements are equal to lam1 and diagonal elements are zeros. 
    If the input lam1 is a matrix, then set its diagonal to be zeros.
    """
    if isinstance(lam1, float) | isinstance(lam1, int):
        lambda1 = ( np.repeat(1.0, p**2).reshape(p, p) - np.identity(p) ) * lam1
    elif isinstance(lam1, np.ndarray):
        lambda1 = lam1 - np.diag( np.diag(lam1) )
    else:
        raise Exception('lam1 should be either a number or an ndarray.')
    return lambda1.reshape(p**2,)



# The g-concord algorithm

def graphical_concord(sam_cov, lam1 = 0.1, lam2 = 0, method = 'ista', 
                      tol = 1e-5, maxit = 100, steptype = 0, assume_scaled = False):
    """
    PseudoNet regularized sparse inverse covariance estimator
    
    Parameters
    ----------
    sam_cov: 2D ndarray (p_features, p_features)
        Sample covariance from which to compute the gconcord.
        
    lam1: non-negative float
        The L1 regularization parameter: the higher the value, the more
        regularization, and the sparser the inverse covariance.
        
    lam2: non-negative float
        The Frobenius norm regularization parameter: the higher the value,
        the more regularization.
        
    method: {'ista', 'fista', 'coordinatewise'}
        The solver to use: iterative soft-thresholding algorithm (ista), 
        fast ISTA (fista), or coordinate-wise descent algorithm (cooordinatewise).
        
    tol: positive float, optional
        The tolerance to declare convergence.
        
    maxit: integer, optional
        The maximum number of iterations.
        
    steptype: {0, 1, 2}, optional
        The type of initial step size used in ista and fista. 0 for heuristic size
        of Barzilai-Borwein, 1 for initial size equal to 1, 2 for a feasible step size
        found from a previous iteration.
        
    assume_scaled: {True, False}, optional
        Whether the input sample covariance matrix is scaled or not.
        
    Returns
    -------
    omega: 2D ndarray, shape (p_features, p_features)
        The estimated sparse precision matrix.
        
    """
    path = _findpath()                   ## find path of shared library
    _, p = sam_cov.shape                 ## data dimensionality
    
    if assume_scaled:                    ## inverse of standard deviation
        sdiv = np.ones(p)
    else:
        sdiv = np.sqrt( 1/np.diag(sam_cov) ) 
        
    SamCor = (sdiv * sam_cov).T * sdiv   ## sample correlation matrix
    s = SamCor.reshape(p**2,)            ## flatten sample correlation

    libcd = npct.load_library("sharedlib.so", path)  ## load shared library file
    libcd.gconcord.restype = None                    ## reset data type
    libcd.gconcord.argtypes = [array_1d_double, c_int, c_int, 
                               array_1d_double, c_double, c_double, c_int, c_int,
                               array_1d_double, array_1d_int, array_1d_int]
    
    lambda1 = _checklam(lam1, p)         ## convert lambda1 into a matrix parameter
    mth = _checkmethod(method)           ## optimization method
    
    out, outi, outj = _createout(p)      ## initial omega for the iterations
    
    libcd.gconcord(s, p, mth, lambda1, lam2, tol, maxit, steptype, out, outi, outj)
    
    omegacor = _createomega(out, outi, outj, p)      ## recover output as a matrix
    omega = (sdiv * omegacor).T * sdiv               ## scaled back by variance
    
    return omega



class GraphicalConcord():
    """
    PseudoNet regularized sparse inverse covariance estimator
    
    Parameters
    ----------
    sam_cov: 2D ndarray (p_features, p_features)
        Sample covariance from which to compute the gconcord.
        
    lam1: non-negative float
        The L1 regularization parameter: the higher the value, the more
        regularization, and the sparser the inverse covariance.
        
    lam2: non-negative float
        The Frobenius norm regularization parameter: the higher the value,
        the more regularization.
        
    method: {'ista', 'fista', 'coordinatewise'}
        The solver to use: iterative soft-thresholding algorithm (ista), 
        fast ISTA (fista), or coordinate-wise descent algorithm (cooordinatewise).
        
    tol: positive float, optional
        The tolerance to declare convergence.
        
    maxit: integer, optional
        The maximum number of iterations.
        
    steptype: {0, 1, 2}, optional
        The type of initial step size used in ista and fista. 0 for heuristic size
        of Barzilai-Borwein, 1 for initial size equal to 1, 2 for a feasible step size
        found from a previous iteration.
        
    Returns
    -------
    omega: 2D ndarray, shape (p_features, p_features)
        The estimated sparse precision matrix.
    """
    
    def __init__(self, lam1 = 0.1, lam2 = 0, method = 'coordinatewise',
                tol = 1e-5, maxit = 100, steptype = 0,
                assume_centered = True, assume_scaled = False):
        self.lam1 = lam1
        self.lam2 = lam2
        self.method = method
        self.tol = tol
        self.maxit = maxit
        self.steptype = steptype
        self.assume_centered = assume_centered
        self.assume_scaled = assume_scaled
        
    def fit(self, X, y = None):
        """Fit the GraphicalConcord model to X.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, p_features)
            Data from which to compute the inverse covariance matrix
        y : (ignored)
        """
        # Covariance does not make sense for a single feature
        X = check_array(X, ensure_min_features=2, ensure_min_samples=2,
                        estimator=self)

        if self.assume_centered:
            Xbar = np.zeros(X.shape[1])
        else:
            Xbar = X.mean(axis = 0)
            
        sam_cov = np.cov(X - Xbar, rowvar = False)
        
        self.omega = graphical_concord(sam_cov, 
                                       lam1 = self.lam1, 
                                       lam2 = self.lam2,
                                       method = self.method, 
                                       tol = self.tol,
                                       maxit = self.maxit,
                                       steptype = self.steptype,
                                       assume_scaled = self.assume_scaled)
        return self
    
    

## Cross-validation with GraphicalConcord
class GraphicalConcordCV(GraphicalConcord):
    """
    """
    
    def __init__(self, lam1s = None, lam2s = None, 
                 random_state = None, shuffle = False, folds = 3,
                 tol = 1e-5, maxit = 100, steptype = 0, method = 'coordinatewise', 
                 assume_centered = True, assume_scaled = False):
        super().__init__(
            lam1 = None, lam2 = None, method = method, tol = tol, maxit = maxit, 
            steptype = steptype, assume_centered = assume_centered,
            assume_scaled = assume_scaled)
        self.lam1s = lam1s
        self.lam2s = lam2s
        self.random_state = random_state
        self.shuffle = shuffle
        self.folds = folds
        
    def _predrisk(self, omega, data):
        """Pridictive risk objective function in cross-validation.
        
        Omega is the inverse covariance matrix estimated from a training set.
        Data is the (n_samples, p_features) 2Darray matrix.
        """
        arg = np.dot( np.dot(data, omega), np.diag(1/np.diag(omega)) )
        return np.linalg.norm(arg, ord = 'fro') / data.shape[0]
    
    def _candidate(self):
        """Determine candidate values for lam1 and lam2
        """
        if self.lam1s is None:
            self.lam1s = 10 ** np.linspace(-1.2, 0, 25) - 10 ** (-1.2)
        if self.lam2s is None:
            self.lam2s = (10 ** np.linspace(-1.2, 0, 25) - 10 ** (-1.2)) * 10
            
        if isinstance(self.lam1s, (int, float)):
            lam1n = 1
            lam1type = 1   # a scalar
        elif isinstance(self.lam1s, np.ndarray) and len( self.lam1s.shape ) > 1:
            lam1n = 1
            lam1type = 2   # a matrix
            self.lam1s = self.lam1s - np.diag( np.diag(self.lam1s) )
        else:
            lam1n = len(self.lam1s)
            lam1type = 3   # a vector
            
        if isinstance(self.lam2s, (float, int)):
            lam2n = 1
            lam2type = 2   # a scalar
        else:
            lam2n = len(self.lam2s)
            lam2type = 3   # a vector
        
        return lam1n, lam2n, lam1type, lam2type
        
        
    def fit(self, X, y = None):
        
        # Covariance does not make sense for a single feature
        x = check_array(X, ensure_min_features = 2, estimator = self)
        n, p = x.shape
    
        kf = KFold(n_splits = self.folds, random_state = self.random_state,
                   shuffle = self.shuffle)
        lam1n, lam2n, lam1type, lam2type = self._candidate()
        
        self.res = []
        
        for i in range(lam1n):
            if lam1type < 3:
                lam1 = self.lam1s
                self.lam1 = lam1
            else:
                lam1 = self.lam1s[i]
            lam1val = []
            
            for j in range(lam2n):
                if lam2type < 3:
                    lam2 = self.lam2s
                    self.lam2 = lam2
                else:
                    lam2 = self.lam2s[j]
                lam2val = []
                
                for train_index, test_index in kf.split(x):
                    sam_cov = np.cov(x[train_index], rowvar = False)
                    omega = graphical_concord(sam_cov = sam_cov, lam1 = lam1, lam2 = lam2, 
                                              method = self.method, tol = self.tol,
                                              maxit = self.maxit, 
                                              steptype = self.steptype,
                                              assume_scaled = self.assume_scaled)
                    cost = self._predrisk(omega, x[test_index])
                    lam2val.append(cost)
                    
                lam1val.append( np.mean(lam2val) )
            
            self.res.append(lam1val)
        
        idx = np.argwhere( self.res == np.min(self.res) )
        if self.lam1 is None:
            self.lam1 = self.lam1s[idx[0][0]]
        if self.lam2 is None:
            self.lam2 = self.lam2s[idx[0][1]]
            
        sam_cov = np.cov(x, rowvar = False)
        self.omega = graphical_concord(sam_cov, lam1 = self.lam1, lam2 = self.lam2, 
                                       method = self.method, tol = self.tol,
                                       maxit = self.maxit, 
                                       steptype = self.steptype,
                                       assume_scaled = self.assume_scaled)
        
        return self
        
            