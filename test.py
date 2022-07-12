from alternative import Alternatives
from exploration import Exploration
import numpy as np

def main(num_alternative=10,num_rounds=100,budget_per_rounds=1,context_list=["standard"]):
    alter=Alternatives(num_alternative=num_alternative,max_mean=(num_alternative-1)/2,noise=False)
    ex=Exploration(alter,num_rounds=num_rounds,context_list=context_list)
    explored_result=ex.collect_data(num_rounds=num_rounds,
                              budget_per_rounds=budget_per_rounds,
                              strategy="random")
    # explored_result=ex.collect_data(num_rounds=num_rounds,
    #                         budget_per_rounds=budget_per_rounds,
    #                         strategy="EVI-based-small-samples")

    explored_mean=np.zeros(shape=(num_alternative,))
    explored_var=np.zeros(shape=(num_alternative,))
    for r in range(num_alternative):
        if np.any(explored_result[r])!=None:
            explored_mean[r]=np.mean(explored_result[r])
            explored_var[r]=np.var(explored_result[r])/np.sqrt(explored_result[r].shape[0])
    print("mean is ",explored_mean)
    print("var is ",explored_var)


if __name__ == "__main__":
    main()