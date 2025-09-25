# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from patsy import dmatrix, build_design_matrices

class MultipleRegression:
    def __init__(self, cohort, data, formula, standardize=[]):
        cohort = cohort.copy()
        self._X_stats = {}
        for c in standardize:
            mu = np.mean(cohort[c])
            sig = np.std(cohort[c], ddof=1)
            cohort[c] = (cohort[c] - mu) /sig
            self._X_stats[c] = (mu, sig)
        self.X = dmatrix(formula, data=cohort, return_type='dataframe')
        self.n, self.p = self.X.shape
        self.predictor_names = list(self.X.columns)
        if hasattr(data, 'columns'):
            self.feature_names = list(data.columns)
        else:
            self.feature_names = list(range(self.p))
        self.y = np.asarray(data)
        self.m = data.shape[1]
        self.fit()

    def fit(self):
        X = np.asarray(self.X, dtype=float)
        XtX_inv = np.linalg.inv(X.T @ X)
        self.beta = XtX_inv @ X.T @ self.y
        self.y_pred = X @ self.beta
        self.residuals = self.y - self.y_pred
        self.sigma_squared = np.sum(self.residuals ** 2) / (self.n - self.p)
        self.var_beta = np.diag(XtX_inv) * self.sigma_squared
        self.se = np.sqrt(self.var_beta)
        self.t_stats = self.beta / self.se
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_stats), df=self.n - self.p))
        crit_t = stats.t.ppf(0.975, df=self.n - self.p)
        self.ci_lower = self.beta - crit_t * self.se
        self.ci_upper = self.beta + crit_t * self.se
        self.standardized_beta = self.beta[1:] * (np.std(X, axis=0) / np.std(self.y))
        self.cohens_d = self.beta / self.se
        # self.cohens_d = 2 * self.t_stats / np.sqrt(self.n - self.p - 1)
        self.fdr_p_values = fdrcorrection(self.p_values)[1]
    
    def get_param_as_dataframe(self, key):
        z = self.__getattr__(key)
        if key == 'sigma_squared':
            if self.m == 1:
                return pd.Series(data=[z], index=self.feature_names)
            else:
                return pd.Series(data=z, index=self.feature_names)
        a,b = z.shape
        if (a,b) == (self.p, self.m):
            return pd.DataFrame(data=z, index=self.predictor_names, columns=self.feature_names)
        elif (a,b) == (self.p - 1, self.m):
            nonintercept_names = [n for n in self.predictor_names if n != 'Intercept']
            return pd.DataFrame(data=z, index=nonintercept_names, columns=self.feature_names)
        elif (a,b) == (self.n, self.m):
            return pd.DataFrame(data=z, index=self.X.index, columns=self.feature_names)
        else:
            raise AttributeError(f'No regression result under {key}')
    
    def summary(self):
#        return {
#            name: {
#                "beta": self.beta[i],
#                "standard_error": self.se[i],
#                "confidence_interval": (self.ci_lower[i], self.ci_upper[i]),
#                "t_statistic": self.t_stats[i],
#                "p_value": self.p_values[i],
#                "fdr_corrected_p_value": self.fdr_corrected_pvals[i],
#                "cohens_d": self.cohens_d[i]
#            }
#            for i, name in enumerate(self.X.columns)
#        }
        return [ self.get_param_as_dataframe(k) for k in ('beta', 'se', 
                'ci_lower', 'ci_upper', 'standardized_beta', 't_stats', 
                'cohens_d', 'p_values', 'fdr_p_values') ]

    def f_test_compare(self, reduced_model):
        if not isinstance(reduced_model, MultipleRegression):
            raise ValueError("Argument must be an instance of MultipleRegression")
        if self.n != reduced_model.n:
            raise ValueError("Both models must have the same number of observations")
        rss_reduced = np.sum(reduced_model.residuals ** 2)
        rss_full = np.sum(self.residuals ** 2)
        df1 = self.p - reduced_model.p
        df2 = self.n - self.p
        f_stat = ((rss_reduced - rss_full) / df1) / (rss_full / df2)
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        return f_stat, p_value
     
    def calculate_residuals(self, other_cohort, other_data):
        other_cohort = other_cohort.copy()
        for c in other_cohort.columns:
            if c in self._X_stats:
                mu, sig = self._X_stats[c]
                other_cohort[c] = (other_cohort[c] - self._)
        X = build_design_matrices([self.X.design_info], other_cohort, return_type='dataframe')[0]
        pred_other = X @ self.beta 
        residuals = other_data - pred_other
        return residuals

