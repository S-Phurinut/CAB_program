import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import math
import seaborn as sns

class Sequential_EVI_algorithm():
    def __init__(self,alternative,num_rounds,budget_per_rounds,context_list,first_stage_sample_size=6,times_of_sampling=1):
        """ Parameters:
            o first_stage_sample_size n0 : the number of time for each alt is played equally
            o additional_total_sample N : the maximum of budget we can play in round of sampling (additional stage) 
        """
        self.alternative=alternative
        self.num_rounds=num_rounds
        self.budget_per_rounds=budget_per_rounds
        self.context_list=context_list

        self.sample_size=first_stage_sample_size
        self.additional_total_sample=self.num_rounds-(self.alternative.num_alt*self.sample_size)
        self.times_of_sampling=times_of_sampling
    
    def EVI_small_strategy(self,alternative,num_rounds,budget_per_rounds,context_list):
        #Setting 
        if alternative==None:
            alter=self.alternative
        else:
            alter=alternative

        first_stage_sample_size=6
        
        additional_total_sample=num_rounds-(alter.num_alt*first_stage_sample_size)
        times_of_sampling=additional_total_sample
        
        num_sample=np.zeros(shape=(alter.num_alt,)) 
        #Sample x_i1,...,x_in0 independently
        for id_stage in range(first_stage_sample_size):
            reward_dict=alter.main_reward()
            if id_stage==0:
                reward=reward_dict
            else:
                for r in range(alter.num_alt):
                    if reward_dict[r]!=None:
                        reward[r]=np.concatenate((reward[r],reward_dict[r]))
        num_sample=num_sample+np.ones(shape=(alter.num_alt,))*first_stage_sample_size
          
        stop=0
        #Determine the sample statistics
        mean_reward=np.zeros(shape=(alter.num_alt,))
        variance_reward=np.zeros(shape=(alter.num_alt,))
        for r in range(alter.num_alt):
            mean_reward[r]=np.mean(reward[r],axis=0)
            variance_reward[r]=np.var(reward[r],axis=0)*first_stage_sample_size/(first_stage_sample_size-1)
 
        #Sort order of mean reward
        order_array_truth=np.arange(0,alter.num_alt) #true order array
        order_array_emp=np.argsort(mean_reward) #get index by sorting mean_reward
#         print("the order of alternative after the first stage of sampling indepently is",1+order_array_emp)
        
        λ_ik=np.zeros(shape=(alter.num_alt,))
        # ν_ik=np.zeros(shape=(alter.num_alt,))
        
#         print("----- allocate a good policy -----")
        need_initialise=True
        num_loop=0
        while stop<(alter.num_alt*first_stage_sample_size)+additional_total_sample:
            initial_alt_set=np.arange(0,alter.num_alt)
            initial_alt_set=set(initial_alt_set) #contain {(1),(2),...,(k)}
            best_alt_id=order_array_emp[-1] #address the index of empirically best index
        
            need_allocation=True
            if need_allocation==True:
                #Calculate necessary parameter for the next step
                d_ik=np.zeros(shape=(alter.num_alt,))
                d_star_ik=np.zeros(shape=(alter.num_alt,))
                for j in initial_alt_set:
                    τ_i=np.zeros((alter.num_alt,))
                    τ_i[j]=additional_total_sample/times_of_sampling
                    λ_ik[j]=1/((variance_reward[order_array_emp[j]]*τ_i[j])/(num_sample[order_array_emp[j]]*(num_sample[order_array_emp[j]]+τ_i[j]))
                                    +(variance_reward[best_alt_id]*τ_i[-1])/(num_sample[best_alt_id]*(num_sample[best_alt_id]+τ_i[-1])))
                    d_ik[j]=mean_reward[best_alt_id]-mean_reward[order_array_emp[j]]
                    d_star_ik[j]=d_ik[j]*np.sqrt(λ_ik[j])
                
                best_EVI_index=0
                best_EVI=-math.inf
                #Compute EVI of each alternative (i)
                for j in initial_alt_set:
                    if j!=alter.num_alt-1:
                        EVI=t.cdf(-d_star_ik[j], df=num_sample[order_array_emp[j]]-1, loc=0, scale=1)
                    else:
                        EVI=t.cdf(-d_star_ik[alter.num_alt-2], df=num_sample[best_alt_id]-1, loc=0, scale=1)
                    
                    if EVI>best_EVI:
                        best_EVI=EVI
                        best_EVI_index=j
                τ_i=np.zeros((alter.num_alt,))
                τ_i[best_EVI_index]=additional_total_sample/times_of_sampling 
                
                sort_index = np.argsort(order_array_emp)
                reorder_τ_i=τ_i[sort_index]
                
                #Step (e): Observe τ_i additional samples
                index=0
                for add_num_stage in τ_i:
                    if add_num_stage!=0 and math.isnan(add_num_stage)==False:
                        for j in range(math.floor(add_num_stage)):
                            index_context=num_loop%len(context_list)
                            reward_dict=alter.context_based_reward(context=context_list[index_context],action=[order_array_emp[index]])
                            for r in range(alter.num_alt):
                                if reward_dict[r]!=None:
                                    reward[r]=np.concatenate((reward[r],reward_dict[r]))
                    index+=1
                num_loop=num_loop+1

                #Update the sample statistics
                num_sample=num_sample+reorder_τ_i
                for r in range(alter.num_alt):
                    mean_reward[r]=np.mean(reward[r],axis=0)
                    variance_reward[r]=np.var(reward[r],axis=0)*num_sample[r]/(num_sample[r]-1)

                #Sort order of mean reward
                order_array_truth=np.arange(0,alter.num_alt) #true order array
                order_array_emp=np.argsort(mean_reward) #get index by sorting mean_reward

                #Update the stopping criteria
                stop=np.sum(num_sample)
#                     print("#Samples we explored so far =",stop)
        
#         print("\n------------------------------------------------------------------")
# #         print("allocation is ",reorder_τ_i)
#         print("The number of play in each alternative i is",num_sample)                 
#         print("The average of reward is ",mean_reward)            
#         print("The best alternative that we obeserved so far is =",order_array_emp[-1]+1)           
        return reward

    