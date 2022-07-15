import numpy as np
import math

class Alternatives():
    def __init__(self,num_alternative=2,max_mean=1,var=1,noise=False):
        self.num_alt=num_alternative
        self.noise=noise
        self.max_mean_alt=max_mean
        self.var_alt=var
        
    def main_reward(self,num_alternative=None,action=None):
        max_mean=self.max_mean_alt
        variance=self.var_alt
        
        if num_alternative==None:
            num_alternative=self.num_alt
        
        keys=list(range(num_alternative))
        reward_dict=dict.fromkeys(keys, None)
        
        if action==None:
            for alt in range(num_alternative):
                
                if self.noise==True:
                    noise=np.random.randn(1,)
                else:
                    noise=0
                    
                #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                reward=(alt)/(num_alternative-1)*max_mean+np.sqrt(variance) * np.random.randn(1,) 
                reward_dict[alt]=reward+noise
        else:
            #action is a list containing the alternative we wanna investigate
            for alt in action:
                
                if self.noise==True:
                    noise=np.random.randn(1,)
                else:
                    noise=0
                    
                #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                reward=(alt)/(num_alternative-1)*max_mean+np.sqrt(variance) * np.random.randn(1,) 
                if  np.any(reward_dict[alt])==None:
                    reward_dict[alt]=reward+noise
                else :
                    reward_dict[alt]=np.concatenate((reward_dict[alt],reward+noise))
        return reward_dict

    def context_based_reward(self,num_alternative=None,context=None,action=None,noise=False,**parameter):
        if num_alternative==None:
            num_alternative=self.num_alt

        mean_shift=0
        variance=self.var_alt
        
        #Define different contexts that could happen 
        if context==None or context=="standard":
            return self.main_reward(num_alternative=num_alternative,action=action)
        
        elif context=="changing-variance":
            variance=2
            for key, value in parameter.items():
                key=value
            #Now, define variance of 2 means high variance
            keys=list(range(num_alternative))
            reward_dict=dict.fromkeys(keys, None)
            if action==None:
                for alt in range(num_alternative):
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=(alt)/(num_alternative-1) +np.sqrt(variance) * np.random.randn(1,)
                    reward_dict[alt]=reward
            
            else:
                for alt in action:
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=(alt)/(num_alternative-1)+np.sqrt(variance) * np.random.randn(1,)
                    if  np.any(reward_dict[alt])==None:
                        reward_dict[alt]=reward+noise
                    else :
                        reward_dict[alt]=np.concatenate((reward_dict[alt],reward+noise))
        
        elif context=="shifting-mean":
            mean_shift=1
            for key, value in parameter.items():
                key=value
            #add constant of 1 to true mean for all alternative
            #then ragnge of mean is tweaked from [0,1] to [0+mean_shift,1+mean_shift]
            keys=list(range(num_alternative))
            reward_dict=dict.fromkeys(keys, None)
            if action==None:
                for alt in range(num_alternative):
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=(alt)/(num_alternative-1) + 1 * np.random.randn(1,) + mean_shift
                    reward_dict[alt]=reward
            
            else:
                for alt in action:
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=(alt)/(num_alternative-1) + 1 * np.random.randn(1,) + mean_shift
                    if  np.any(reward_dict[alt])==None:
                        reward_dict[alt]=reward+noise
                    else :
                        reward_dict[alt]=np.concatenate((reward_dict[alt],reward+noise))

                    
        elif context=="higher-mean-lower-variance":
            #Now, increase varaince when mean is low linearly (mean<=0.5,variance>=1)
            #by determinning that mean=0 var=2 --> mean=1 var=1
            keys=list(range(num_alternative))
            reward_dict=dict.fromkeys(keys, None)
            if action==None:
                for alt in range(num_alternative):
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=(alt)/(num_alternative-1) + (2*(1-(alt)/(num_alternative-1))
                                                      +0.5*((alt)/(num_alternative-1))) * np.random.randn(1)
                    reward_dict[alt]=reward
            
            else:
                for alt in action:
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=(alt)/(num_alternative-1) + (2*(1-(alt)/(num_alternative-1))
                                                      +0.5*((alt)/(num_alternative-1))) * np.random.randn(1)
                    if  np.any(reward_dict[alt])==None:
                        reward_dict[alt]=reward+noise
                    else :
                        reward_dict[alt]=np.concatenate((reward_dict[alt],reward+noise))
                    

            
        elif context=="reorder-mean":
            #swap order of alternative from [0,1] to [1,0]
            keys=list(range(num_alternative))
            reward_dict=dict.fromkeys(keys, None)
            if action==None:
                for alt in range(num_alternative):
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=1-(alt)/(num_alternative-1) + 1 * np.random.randn(1,)
                    reward_dict[alt]=reward
            
            else:
                for alt in action:
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=1-(alt)/(num_alternative-1) + 1 * np.random.randn(1,)
                    if  np.any(reward_dict[alt])==None:
                        reward_dict[alt]=reward+noise
                    else :
                        reward_dict[alt]=np.concatenate((reward_dict[alt],reward+noise))
                    

        
        elif context=="exceptional-mean":
            #add the accessive value to true mean (in case situaton is exceptional)
            keys=list(range(num_alternative))
            reward_dict=dict.fromkeys(keys, None)
            if action==None:
                for alt in range(num_alternative):
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0 
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=5+(alt)/(num_alternative-1) + 1 * np.random.randn(1,)
                    reward_dict[alt]=reward
            
            else:
                for alt in action:
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=5+(alt)/(num_alternative-1) + 1 * np.random.randn(1,)
                    if  np.any(reward_dict[alt])==None:
                        reward_dict[alt]=reward+noise
                    else :
                        reward_dict[alt]=np.concatenate((reward_dict[alt],reward+noise))

                    
        elif context=="broken-mean":
            #add the accessive value to true mean (in case situaton is exceptional)
            keys=list(range(num_alternative))
            reward_dict=dict.fromkeys(keys, None)
            if action==None:
                for alt in range(num_alternative):
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=100+(alt)/(num_alternative-1) + 1 * np.random.randn(1,)
                    reward_dict[alt]=reward
            
            else:
                for alt in action:
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=100+(alt)/(num_alternative-1) + 1 * np.random.randn(1,)
                    if  np.any(reward_dict[alt])==None:
                        reward_dict[alt]=reward+noise
                    else :
                        reward_dict[alt]=np.concatenate((reward_dict[alt],reward+noise))

        elif context=="collapsed-mean":
            #add the accessive value to true mean (in case situaton is exceptional)
            keys=list(range(num_alternative))
            reward_dict=dict.fromkeys(keys, None)
            if action==None:
                for alt in range(num_alternative):
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=(alt)/(num_alternative-1) + 1 * np.random.randn(1,)-100
                    reward_dict[alt]=reward
            
            else:
                for alt in action:
                    if self.noise==True:
                        noise=np.random.randn(1,)
                    else:
                        noise=0
                    #Reward distribution is endowed by normal dis. N(μ,σ^2) 
                    reward=(alt)/(num_alternative-1) + 1 * np.random.randn(1,)-100
                    if  np.any(reward_dict[alt])==None:
                        reward_dict[alt]=reward+noise
                    else :
                        reward_dict[alt]=np.concatenate((reward_dict[alt],reward+noise))

        return reward_dict