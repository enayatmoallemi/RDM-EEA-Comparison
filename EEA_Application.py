
# coding: utf-8

# In[1]:

import sys
sys.path.append(r'C:\Users\z3521150\Google Drive\My Library\UNSW\EMA\EMAworkbench-master\src')

#from ema_workbench.em_framework import (ModelEnsemble, ParameterUncertainty, Outcome)
from ema_workbench.util import ema_logging
from ema_workbench.connectors import import_anylogic_csv_eea
from ema_workbench.analysis import plotting
from ema_workbench.analysis.plotting_util import BOXPLOT, KDE, VIOLIN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns
from ema_workbench.analysis import prim
from ema_workbench import notebook_kernel_config


# In[4]:

uncertainties = ['risk of loss 0', 'life time 0', 'roe 0', 'time in cap 0', 'time between cap 0', 
                 'time in dm 0', 'time between dm 0', 'time in om 0', 'time between om 0', 'cost om', 'number purchased fleets', 
                 'number allowed cap', 'number allowed om', 'number allowed dm']

outcomes_of_interest = ['in_service', 'maintenance_cost', 'om_queue_0', 'dm_queue_0', 'aging_nc_0', 'aging_ma_0', 
                        'aging_oc_0', 'acquisition_cost', 'average_flying_hours']

policies = [{'name':'epoch_1', 
             'file':r'C:\Users\z3521150\Google Drive\My Library\UNSW\Publications\Epoch-Era Analysis\data\500_exp_era1_epoch1'}, 
            {'name':'epoch_2', 
             'file':r'C:\Users\z3521150\Google Drive\My Library\UNSW\Publications\Epoch-Era Analysis\data\500_exp_era1_epoch2'},
            {'name':'epoch_3', 
             'file':r'C:\Users\z3521150\Google Drive\My Library\UNSW\Publications\Epoch-Era Analysis\data\500_exp_era1_epoch3' }, 
            {'name':'epoch_4', 
             'file':r'C:\Users\z3521150\Google Drive\My Library\UNSW\Publications\Epoch-Era Analysis\data\500_exp_era1_epoch4' }, 
            {'name':'epoch_5', 
             'file':r'C:\Users\z3521150\Google Drive\My Library\UNSW\Publications\Epoch-Era Analysis\data\500_exp_era1_epoch5' }]

number_of_experiments = 500

era_duration = 800

x = import_anylogic_csv_eea.ImportfromAnyLogicEEA(uncertainties, outcomes_of_interest, policies, number_of_experiments, era_duration)


results = x.extract_results()


# In[292]:

experiments, outcomes = results


# In[293]:


def pareto_frontier(new_exp_x, new_exp_y, maxX = True, maxY = True):
# Sort the list in either ascending or descending order of X
    
    
    myList = sorted([[new_exp_x[i], new_exp_y[i]] for i in range(len(new_exp_x))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY: 
            if pair[1][0] >= p_front[-1][1][0]: # Look for higher values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
        else:
            if pair[1][0] <= p_front[-1][1][0]: # Look for lower values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
# Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0][0] for pair in p_front]
    p_frontY = [pair[1][0] for pair in p_front]
    p_front_acq = [pair[0][1] for pair in p_front]
    p_front_om = [pair[0][2] for pair in p_front]
    p_front_dm = [pair[0][3] for pair in p_front]
    p_front_cap = [pair[0][4] for pair in p_front]

    return p_frontX, p_frontY, p_front_acq, p_front_om, p_front_dm, p_front_cap



# In[294]:

get_ipython().magic(u'matplotlib inline')
import mpld3


data_1 = outcomes['average_flying_hours'][:, -1]
data_2 = outcomes['maintenance_cost'][:, -1]+ outcomes['acquisition_cost'][:, -1]

policies_to_compare = np.unique(experiments['policy'])


exp = pd.DataFrame(experiments)
out_1 = pd.DataFrame(data_1)
out_2 = pd.DataFrame(data_2)

para_list = ['number purchased fleets', 'number allowed om', 'number allowed dm', 'number allowed cap']

design = {}
for i, para in enumerate(para_list):
    design[i] = pd.DataFrame(experiments[para])


    
fig, ax = plt.subplots(1, 5, figsize=(28, 5))

for i, policy in enumerate(policies_to_compare):
    
    out_array_1 = out_1[exp['policy'] == policy].values
    out_array_2 = out_2[exp['policy'] == policy].values
    out_list_1 = out_array_1.tolist()
    out_list_2 = out_array_2.tolist()
    
    exp_array = {}
    exp_list = {}
    for k in range(len(design)):
        exp_array[k] = design[k][exp['policy'] == policy].values
        exp_list[k] = exp_array[k].tolist()
   

    
    new_exp_x = [list(t) for t in zip(out_list_2, exp_list[0], exp_list[1], exp_list[2], exp_list[3])]
    new_exp_y = [list(t) for t in zip(out_list_1, exp_list[0], exp_list[1], exp_list[2], exp_list[3])]
    

    # get your data from somewhere to go here
    # Find lowest values for cost and highest for savings
    p_front = pareto_frontier(new_exp_x, new_exp_y, maxX = False, maxY = True) 
    # Plot a scatter graph of all results
    

    for j in range(len(new_exp_x)):
        if ((new_exp_x[j][1] >= [4]) and (new_exp_x[j][2] <= [3]) and (new_exp_x[j][3] <= [3])):
            ax[i].scatter(new_exp_x[j][0], new_exp_y[j][0], color='red')
        elif ((new_exp_x[j][1] <= [3]) and (new_exp_x[j][2] > [3]) and (new_exp_x[j][3] > [3])):
            ax[i].scatter(new_exp_x[j][0], new_exp_y[j][0], color='steelblue')
        #elif ((new_exp_x[j][1] <= [2]) ):
            #ax[i].scatter(new_exp_x[j][0], new_exp_y[j][0], color='green')
        else:
            ax[i].scatter(new_exp_x[j][0], new_exp_y[j][0], color='grey')
            
    ax[i].tick_params(axis='both', labelsize=15)
    



            
    # Then plot the Pareto frontier on top
    ax[i].plot(p_front[0], p_front[1], color='black')
    xlist = []
    ylist = []
    for x, y in zip(p_front[0], p_front[1]):
        xlist.append(x[0])
        ylist.append(y[0])
    
    labels = []
    for a, b, c, d in zip(p_front[2], p_front[3], p_front[4], p_front[5]):
        labels.append("(" + str(a).strip('[]' + 'L') + ", " + str(b).strip('[]'+ 'L') + ", " + str(c).strip('[]'+ 'L') + ", " + str(d).strip('[]'+ 'L') + ")")

    print "epoch" + str(i)    
    for k, txt in enumerate(labels):
        pareto_points = "design:" + str(txt) + " cost:" + str(xlist[k]) + " flyinghour:" + str(ylist[k])    
        print pareto_points
        
    #for k, txt in enumerate(labels):
        #ax[i].annotate(txt, (xlist[k],ylist[k]))
        #footnote = "design:" + str(txt) + " cost:" + str(xlist[k]) + " flyinghour:" + str(ylist[k])
        #ax[i].annotate(footnote, xy= (1,-(k*25+70)), xycoords='axes points', size=20)
     

fig.text( .5, .005, 'Total maintenance and acquisition cost ($ billion)', ha='center', fontsize=19)
fig.text( .1, .84, 'Total average flying hours (hour)', ha='center', rotation='vertical', fontsize=19)



plt.savefig('{}/fig{}.png'.format(r'C:\Users\z3521150\Google Drive\My Library\UNSW\Papers\Epoch-era analysis\figs', 
                                    'EEA1'), dpi=300, bbox_inches='tight')


# In[240]:

get_ipython().magic(u'pinfo plt.annotate')

