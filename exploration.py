from alg_random import random_strategy
from alg_EVI_based import Sequential_EVI_algorithm
import numpy as np

class Exploration():
    def __init__(self,alternative,num_rounds=1,budget_per_rounds=1,context_list=[None],**kwargs):
        self.alternative=alternative
        self.context_list=context_list
        self.num_rounds=num_rounds
        self.budget=budget_per_rounds
    
    def collect_data(self,num_rounds=None,budget_per_rounds=None,strategy="random",context_list=[None]):
        """return output in a dict of reward where key is an alternative"""
        if context_list==[None]:
            context_list=self.context_list
        if num_rounds==None:
            num_rounds=self.num_rounds
        if budget_per_rounds==None:
            budget_per_rounds=self.budget
        if budget_per_rounds>self.alternative.num_alt:
            raise Exception("budget_per_rounds must be less than num_alt")
        
        if strategy=="random":    
            return random_strategy(self.alternative,num_rounds,budget_per_rounds,context_list)

        if strategy=="EVI-based-small-samples":
            salg=Sequential_EVI_algorithm(self.alternative,num_rounds,budget_per_rounds,context_list)
            return salg.EVI_small_strategy(self.alternative,num_rounds,budget_per_rounds,context_list)
        
        if strategy=="epsilon_greedy":
            epsilon=0.8 #Set probability to choose the same alteenative in the next rounds
            
            reward=np.array([1,2],[3,4])
            return reward