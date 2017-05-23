
# coding: utf-8

# In[1]:

import sys
sys.path.append(r'C:\Users\z3521150\Google Drive\My Library\UNSW\EMA\EMAworkbench-master\src')

#from ema_workbench.em_framework import (ModelEnsemble, ParameterUncertainty, Outcome)
from ema_workbench.util import ema_logging
from ema_workbench.connectors import import_anylogic_csv
from ema_workbench.connectors import import_anylogic_csv_nopolicy
from ema_workbench.analysis import plotting
from ema_workbench.analysis.plotting_util import BOXPLOT, KDE, VIOLIN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns
from ema_workbench.analysis import prim
from ema_workbench import notebook_kernel_config


# In[3]:

uncertainties = ['risk of loss 0', 'life time 0', 'roe 0', 'time in cap 0', 'time between cap 0', 
                 'time in dm 0', 'time between dm 0', 'time in om 0', 'time between om 0', 'cost om', 'number purchased fleets', 
                 'number allowed cap', 'number allowed om', 'number allowed dm']

outcomes_of_interest = ['in_service', 'maintenance_cost', 'om_queue_0', 'dm_queue_0', 'aging_nc_0', 'aging_ma_0', 
                        'aging_oc_0', 'acquisition_cost', 'average_flying_hours']

policies = [{'name':'High Acquisition-Low Maintenance', 
             'file':r'C:\Users\z3521150\Google Drive\My Library\UNSW\Publications\Epoch-Era Analysis\data\500_exp_6A0C1O1D'}, 
            {'name':'Low Acquisition-High Maintenance', 
             'file':r'C:\Users\z3521150\Google Drive\My Library\UNSW\Publications\Epoch-Era Analysis\data\500_exp_2A3C3O3D'},
            {'name':'Medium Acquisition-Medium Maintenance', 
             'file':r'C:\Users\z3521150\Google Drive\My Library\UNSW\Publications\Epoch-Era Analysis\data\500_exp_4A1C2O2D' }]

number_of_experiments = 500

x = import_anylogic_csv.ImportfromAnyLogicPolicy(uncertainties, outcomes_of_interest, policies, number_of_experiments)

results = x.extract_results()




# In[4]:

experiments, outcomes = results


# ## Boxplots for Comparing Different Policies

# In[5]:

get_ipython().magic(u'matplotlib inline')

experiments, outcomes = results

ooi = 'average_flying_hours'
data = outcomes[ooi][:, 800]
#data = outcomes['maintenance_cost'][:, 800]+outcomes['acquisition_cost'][:, 800]

policies_to_compare = np.unique(experiments['policy'])


exp = pd.DataFrame(experiments)
out = pd.DataFrame(data)


bplist = []
for policy in policies_to_compare:
    bplist.append(out[exp['policy'] == policy].values)

fig, ax = plt.subplots(1,1)

bp = ax.boxplot(bplist)
xtickNames = plt.setp(ax, xticklabels=['High Acquisition- \n Low Maintenance', 'Low Acquisition- \n High Maintenance', 
                                       'Medium Acquisition- \n Medium Maintenance'])
plt.setp(xtickNames, rotation=0, fontsize=10)

plt.yticks (fontsize = 10)

for box in bp['boxes']:
    box.set(color='k', linewidth=1)
    
for whisker in bp['whiskers']:
    whisker.set(color='k', linewidth=1, linestyle='--')
    
for median in bp['medians']:
    median.set(color='red', linewidth=1)


ax.set_ylabel('Average flying hours (hour)', fontsize=10)

plt.locator_params(nbins=4, axis='y')


plt.savefig('{}/fig{}.png'.format(r'C:\Users\z3521150\Google Drive\My Library\UNSW\Papers\Epoch-era analysis\figs', '_boxplot_flyinghour'), 
            dpi=300, bbox_inches='tight')


# In[12]:

get_ipython().magic(u'matplotlib inline')

experiments, outcomes = results


data = outcomes['maintenance_cost'][:, 800]+outcomes['acquisition_cost'][:, 800]

policies_to_compare = np.unique(experiments['policy'])


exp = pd.DataFrame(experiments)
out = pd.DataFrame(data)


bplist = []
for policy in policies_to_compare:
    bplist.append(out[exp['policy'] == policy].values)

fig, ax = plt.subplots(1,1)

bp = ax.boxplot(bplist)
xtickNames = plt.setp(ax, xticklabels=['High Acquisition- \n Low Maintenance', 'Low Acquisition- \n High Maintenance', 
                                       'Medium Acquisition- \n Medium Maintenance'])
plt.setp(xtickNames, rotation=0, fontsize=10)

plt.yticks (fontsize = 10)

for box in bp['boxes']:
    box.set(color='k', linewidth=1)
    
for whisker in bp['whiskers']:
    whisker.set(color='k', linewidth=1, linestyle='--')
    
for median in bp['medians']:
    median.set(color='red', linewidth=1)

ax.set_ylabel('Total maintenance and acquisition cost ($ billion)', fontsize=10)

plt.locator_params(nbins=4, axis='y')


plt.savefig('{}/fig{}.png'.format(r'C:\Users\z3521150\Google Drive\My Library\UNSW\Papers\Epoch-era analysis\figs', '_boxplot_cost'), 
            dpi=300, bbox_inches='tight')


# In[33]:


def pareto_frontier(out_x, out_y, maxX = True, maxY = True):
# Sort the list in either ascending or descending order of X
    
    
    myList = sorted([[out_x[i], out_y[i]] for i in range(len(out_x))], reverse=maxX)
    # Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
    # Loop through the sorted list
    for pair in myList[1:]:
        if maxY: 
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
    # Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    #p_front_acq = [pair[0][1] for pair in p_front]
    #p_front_om = [pair[0][2] for pair in p_front]
    #p_front_dm = [pair[0][3] for pair in p_front]
    #p_front_cap = [pair[0][4] for pair in p_front]

    return p_frontX, p_frontY#, p_front_acq, p_front_om, p_front_dm, p_front_cap



# In[51]:

get_ipython().magic(u'matplotlib inline')
import mpld3
import matplotlib.patches as mpatches

data_1 = outcomes['average_flying_hours'][:, -1]
data_2 = outcomes['maintenance_cost'][:, -1]+ outcomes['acquisition_cost'][:, -1]

policies_to_compare = np.unique(experiments['policy'])


    
exp = pd.DataFrame(experiments)
out_1 = pd.DataFrame(data_1)
out_2 = pd.DataFrame(data_2)
  
#para_list = ['number purchased fleets', 'number allowed om', 'number allowed dm', 'number allowed cap']

#design = {}
#for i, para in enumerate(para_list):
    #design[i] = pd.DataFrame(experiments[para])


    
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# Then plot the Pareto frontier on top

#transform single column dataframe to a list
out_1_list = out_1[0].tolist()
out_2_list = out_2[0].tolist()

p_front = pareto_frontier(out_2_list, out_1_list, maxX = False, maxY = True) 

ax.plot(p_front[0], p_front[1], color='black')

    
    

for i, policy in enumerate(policies_to_compare):
    

    out_array_1 = out_1[exp['policy'] == policy].values
    out_array_2 = out_2[exp['policy'] == policy].values
    out_list_1 = out_array_1.tolist()
    out_list_2 = out_array_2.tolist()
    
    
      
    
    # Plot a scatter graph of all results
    for j in range(len(out_list_2)):
        if policy == 'High Acquisition-Low Maintenance':
            ax.scatter(out_list_2[j], out_list_1[j], color='red', s=6)
        elif policy == 'Low Acquisition-High Maintenance':
            ax.scatter(out_list_2[j], out_list_1[j], color='steelblue', s=6)
        else:
            ax.scatter(out_list_2[j], out_list_1[j], color='green', s=6)
            
    ax.tick_params(axis='both', labelsize=15)
    

ax.set_xlabel('Total maintenance and acquisition cost ($ billion)', fontsize=10)
ax.set_ylabel('Total average flying hours (hour)', fontsize=10)

plt.xticks (fontsize = 10)
plt.ylim([0, 30000])
plt.yticks (fontsize = 10)
            
red_patch = mpatches.Patch(color='red', label='High Acquisition-Low Maintenance')
green_patch = mpatches.Patch(color='green', label='Medium Acquisition-Medium Maintenance')
blue_patch = mpatches.Patch(color='steelblue', label='Low Acquisition-High Maintenance')

plt.legend(handles=[red_patch, green_patch, blue_patch], bbox_to_anchor=(0., 1, 1., .102), loc=3)


plt.savefig('{}/fig{}.png'.format(r'C:\Users\z3521150\Google Drive\My Library\UNSW\Publications\Epoch-Era Analysis\figs', 
                                    'RDM_scatter'), dpi=300, bbox_inches='tight')

