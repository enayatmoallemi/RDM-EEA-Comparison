
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import os


# In[18]:

class ImportfromAnyLogicPolicy:
    
    def __init__(self, uncertainties, outcomes, policies, number_of_experiments):
        self.uncertainties = uncertainties
        self.outcomes = outcomes
        self.policies = policies
        self.nexp = number_of_experiments

       
    def extract_results(self):
        exp = self.extract_experiments()
        out = self.extract_outcomes()
        results = exp, out
        return results
        
    def extract_experiments(self):
        experiments = np.zeros(self.nexp*len(self.policies), dtype={'names':self.uncertainties, 'formats':['float64']*len(self.uncertainties)})
        for i, val in enumerate(self.policies):
            path = os.path.join(str(self.policies[i]['file']), "parameters.csv")
            db_experiments = pd.read_csv(path)     
            count_policy_exp = i*self.nexp            
            for j in range(self.nexp):
                l = []
                for uncertainty in self.uncertainties:
                    cell = db_experiments.ix[j, uncertainty]
                    l.append(cell)
                unc_set = tuple(l)
                experiments[j+count_policy_exp] = unc_set

        return experiments
      
    def extract_outcomes(self):
        outcomes_dict = {}
        for outcome in self.outcomes:
            for i, val in enumerate(self.policies):
                path = os.path.join(str(self.policies[i]['file']), str(outcome)+".csv")
                df_time_outcome = pd.read_csv(path)
                df_outcome = df_time_outcome.iloc[:, 1:]
                if i == 0:
                    outcomes_dict[outcome] = df_outcome.T.values
                else:
                    outcomes_dict[outcome] = np.vstack([outcomes_dict[outcome], df_outcome.T.values])
        
        outcomes_dict['time'] = df_time_outcome['time'].values
        for i in range(self.nexp*len(self.policies)-1):
            outcomes_dict['time'] = np.vstack( [outcomes_dict['time'] , df_time_outcome['time'].values ] )
        outcomes = outcomes_dict
        return outcomes


# In[ ]:



