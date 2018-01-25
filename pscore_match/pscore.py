"""
This module contains a class to estimate propensity scores.
"""

from __future__ import division
import numpy as np
import scipy
from scipy.stats import binom, hypergeom, gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

################################################################################
##################### Base Propensity Score Class ##############################
################################################################################

class PropensityScore(object):
    """
    Estimate the propensity score for each observation.
    
    The compute method uses a generalized linear model to regress treatment on covariates to estimate the propensity score. 
    This is not the only way to estimate the propensity score, but it is the most common.
    The two options allowed are logistic regression and probit regression.
    """

    def __init__(self, treatment, covariates):
        """
        Parameters
        -----------        
        treatment : array-like
            binary treatment assignment
        covariates : pd.DataFrame
            covariates, one row for each observation
        """
        assert treatment.shape[0]==covariates.shape[0], 'Number of observations in \
            treated and covariates doesnt match'
        self.treatment = treatment
        self.covariates = covariates
        
    def compute(self, method='logistic'):
        """
        Compute propensity score and measures of goodness-of-fit
        
        Parameters
        ----------
        method : str
            Propensity score estimation method. Either 'logistic' or 'probit'
        """
        predictions = None
        if method == 'logistic':
            # i've had a ton of issues w/ the default SM solver and w/ others (including bfgs, which I thought was
            # working but then started giving me 0.5 for everything) - for ex, singular matrix errors, etc. I don't
            # need the things like p-values that SM gives over sklearn, but I do want a more robust implementation,
            # so I'm going to switch to sklearn for here at least
            # there's some useful stuff at https://stackoverflow.com/questions/24924755/logit-estimator-in-statsmodels-and-sklearn

            # high C value means to regularize hardly at all - i'm not standardizing the data so i don't want to
            # drop features incorrectly because of different scales (some regularization is needed because of how the
            # solver works)
            lr = LogisticRegression(C=1e9, fit_intercept=True)
            lr.fit(self.covariates, self.treatment)
            self.model = lr
            predictions = lr.predict_proba(self.covariates)[:,1] # index 1 because we want the prob of a 1

            # old
            #model = sm.Logit(self.treatment, predictors).fit_regularized(alpha = 0.001, disp=False, warn_convergence=True)
            #model = sm.Logit(self.treatment, predictors).fit(method='bfgs', disp=False, warn_convergence=True)
            #model = sm.Logit(self.treatment, predictors).fit(disp=False, warn_convergence=True)
            #model = sm.Logit(self.treatment, predictors).fit(disp=True, warn_convergence=True, maxiter=500)
        elif method == 'probit':
            predictors = sm.add_constant(self.covariates, prepend=False)
            model = sm.Probit(self.treatment, predictors).fit(disp=False, warn_convergence=True)
            self.model = model
            predictions = model.predict()
        else:
            raise ValueError('Unrecognized method')

        return predictions;
