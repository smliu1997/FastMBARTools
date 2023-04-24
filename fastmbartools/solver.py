import numpy as np
import pandas as pd
from .fastmbar import FastMBAR
import fastmbartools.bias as bias
import sys
import os

__author__ = 'Shuming Liu'

bias_dict = {'linear': bias.linear_bias, 
             'harmonic': bias.harmonic_bias}

class FastMBARSolver(object):
    """
    The class for efficiently solving MBAR equations rapidly with FastMBAR. 
    """
    def __init__(self, T, cv, bias_param, bias_func='harmonic', kB=8.314e-3, unbiased_energy=None):
        """
        Initialize. 
        
        Parameters
        ----------
        T : float or int or 1d array like
            Temperatures for the trajectories. 
            If temperatures is float or int, all the simulations share the temperature. 
            If temperatures is array like, it gives the temperature for each simulation. 
        
        cv : array like, cv[i] is 1d or 2d array like
            Collective variables of trajectories. 
            cv[i] includes samples from the i-th state (state means given temperature and bias).
            len(cv) should be the number of states. 
            1d cv[i] is only available if target collective variable is 1d.
            For general cases, cv[i] is 2d. 
        
        bias_param : array like, bias_param[i] should be 1d array like
            Parameters for the bias. len(bias_param) should be the number of states. 
        
        bias_func : str or function or list of functions
            Bias function. 
            If bias_type is string, it has to be 'linear' or 'harmonic', which means all the potentials are linear or harmonic.
            If bias_type is a function, this function is applied to every state. 
            If bias_type is a list, it provides potentials for all the states. Note FastMBAR is versatile and each state can have a different functional form. 
        
        kB : float or int
            Boltzmann constant for the given temperature unit and energy unit.
            For example, if temperature in K, and energy in kJ/mol, then kB should be 0.008314. 
        
        unbiased_energy : None or array like, unbiased_energy[i] is 1d array like
            The unbiased energy of the samples. 
            If None, unbiased energy is not provided. This requires all the simulations and target PMF to be at the same temperature!
            If array like, unbiased_energy[i] is the unbiased energy of samples from the i-th state. 
        
        """
        self.M = len(cv) # number of states
        if np.isscalar(T):
            T = np.array([T]*self.M)
        else:
            assert len(T) == self.M
        self.T = T
        self.n_samples = np.array([len(each) for each in cv]).astype(np.float64)
        X = []
        for i in range(self.M):
            for j in cv[i]:
                X.append(j)
        X = np.array(X)
        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))
        self.X = X # put all the samples together
        assert self.X.ndim == 2
        assert len(bias_param) == self.M
        self.bias_param = bias_param
        if isinstance(bias_func, str):
            self.bias_func = []
            for i in range(self.M):
                self.bias_func.append(bias_dict[bias_func])
        elif isinstance(bias_func, list):
            self.bias_func = bias_func
        else:
            self.bias_func = [bias_func]*self.M
        self.kB = kB
        if unbiased_energy is None:
            self.unique_T = True
            assert np.all(self.T == self.T[0])
            self.E0 = 0
        else:
            self.unique_T = False
            assert len(unbiased_energy) == self.M
            E0 = []
            for i in range(self.M):
                assert len(unbiased_energy[i]) == len(cv[i])
                for j in unbiased_energy[i]:
                    E0.append(j)
            self.E0 = np.array(E0)
            
    def solve(self, bootstrap=True, cuda=False, verbose=False):
        """
        Solve MBAR equations. 
        
        Parameters
        ----------
        bootstrap : bool
        
        cuda : bool
        
        verbose : bool
        
        """
        A = np.zeros((self.M, self.X.shape[0])) + self.E0
        for i in range(self.M):
            A[i, :] += self.bias_func[i](self.X, self.bias_param[i])
            A[i, :] /= (self.kB*self.T[i])
        self.A = A
        self.F = FastMBAR(energy=self.A, num_conf=self.n_samples, cuda=cuda, bootstrap=bootstrap, verbose=verbose)
    
    def compute_pmf(self, target_T, target_cv, bins, min_PMF_as_zero=True):
        """
        Compute PMF along the target collective variables. 
        This only supports computing 1d or 2d or 3d target collective variables. 
        
        Parameters
        ----------
        target_T : float or int
            Target temperature.
        
        target_cv : array like, target_cv[i] is 1d or 2d array like
            Target collective variables of trajectories. 
        
        bins : array like, bins[i] is 1d numpy array
            bins[i] is the bin edges along the i-th dimension.
        
        min_PMF_as_zero : bool
            Whether to set the minimal PMF as zero. 
        
        Returns
        -------
        result : pd.DataFrame
            Output result including PMF and standard deviation of PMF. 
        
        """
        if self.unique_T:
            assert target_T == self.T[0]
        assert len(target_cv) == self.M
        Y = []
        for i in range(self.M):
            assert len(target_cv[i]) == self.n_samples[i]
            for j in target_cv[i]:
                Y.append(j)
        Y = np.array(Y)
        if Y.ndim == 1:
            Y = np.reshape(Y, (-1, 1))
        d = Y.shape[1]
        assert len(bins) == d
        L = 1
        for each in bins:
            L *= (len(each) - 1)
        centers = []
        for each in bins:
            centers.append(0.5*(each[1:] + each[:-1]))
        ext_centers = []
        if d == 1:
            for i in centers[0]:
                ext_centers.append([i])
        elif d == 2:
            for i in centers[0]:
                for j in centers[1]:
                    ext_centers.append([i, j])
        elif d == 3:
            for i in centers[0]:
                for j in centers[1]:
                    for k in centers[2]:
                        ext_centers.append([i, j, k])
        else:
            sys.exit('target_cv dimension is larger than 3.')
        ext_centers = np.array(ext_centers)
        assert ext_centers.ndim == 2
        R = np.zeros((L, Y.shape[0]))
        for i in range(Y.shape[0]):
            H_i, _ = np.histogramdd(Y[[i]], bins)
            H_i[H_i == 0] = np.inf
            H_i[H_i == 1] = 0
            R[:, i] += np.reshape(H_i, -1)
        B = (self.E0 + R)/(self.kB*target_T)
        PMF, PMF_std = self.F.calculate_free_energies_of_perturbed_states(B)
        if min_PMF_as_zero:
            PMF -= np.amin(PMF[~np.isnan(PMF)])
        coord_columns = ['x', 'y', 'z'][:d]
        columns = coord_columns[:d] + ['PMF (kT)', 'PMF_std (kT)']
        result = pd.DataFrame(columns=columns)
        result[coord_columns] = ext_centers
        result['PMF (kT)'] = PMF
        result['PMF_std (kT)'] = PMF_std
        return result


