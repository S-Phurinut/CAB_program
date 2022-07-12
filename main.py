from alternative import Alternatives
from exploration import Exploration
import numpy as np
import os

def main(num_alternative,num_rounds_min,num_rounds_max,step,budget_per_rounds=1,context_list=["standard"],num_simulation=100,name_result="percentage"):
    alter=Alternatives(num_alternative=num_alternative,max_mean=(num_alternative-1)/2,noise=False)
    num_rounds_list=np.arange(num_rounds_min,num_rounds_max+step,step)

    correct_percentage=[]
    for num_rounds in num_rounds_list:
        count=np.zeros((1,num_alternative))
        ex=Exploration(alter,num_rounds=num_rounds,budget_per_rounds=budget_per_rounds,context_list=context_list)
    
        for sim in range(0,num_simulation):
            #Store reward matrix from exploration phase
            explored_result=ex.collect_data(num_rounds=num_rounds,
                                    budget_per_rounds=budget_per_rounds,
                                    strategy="random")

            explored_mean=np.zeros(shape=(num_alternative,))
        #     explored_var=np.zeros(shape=(num_alternative,))
            for r in range(num_alternative):
                if np.any(explored_result[r])!=None:
                    explored_mean[r]=np.mean(explored_result[r])
        #             explored_var[r]=np.var(explored_result[r])

            max_id=np.argmax(explored_mean)
            count[0,max_id]=count[0,max_id]+1
    
        correct_percentage.append(count[0,num_alternative-1]/num_simulation)
    correct_percentage=np.array(correct_percentage)
    print('\n ---- Collect all features to file ----')
    name_result=str(name_result)
    os.chdir("/Users/lightun/Documents/PYTHON learning/AB-testing/run_code/results")
    np.savetxt(name_result, correct_percentage)



if __name__ == "__main__":
    num_alternative=10
    num_rounds_min=num_alternative*2
    num_rounds_max=30
    step=10
    budget_per_rounds=1
    context_list=["standard"]
    num_simualtion=10
    name_result="standard_only"
    main(num_alternative,num_rounds_min,num_rounds_max,step,budget_per_rounds,context_list,num_simualtion,name_result)