import numpy as np

def random_strategy(alternative,num_rounds,budget_per_rounds,context_list):
    for rounds in range(0,num_rounds):
        action=np.random.randint(alternative.num_alt,size=(budget_per_rounds,),dtype=int)
        action=list(action)
        for context in context_list:
            reward_dict=alternative.context_based_reward(
                num_alternative=alternative.num_alt,context=context,action=action)
            if rounds==0 and context==context_list[0]:
                reward=reward_dict
            else:
                for r in range(alternative.num_alt):
                    if np.any(reward_dict[r])!=None and np.any(reward[r])!=None :
                        reward[r]=np.concatenate((reward[r],reward_dict[r]))
                    elif np.any(reward_dict[r])!=None and np.any(reward[r])==None:
                        reward[r]=reward_dict[r]
    return reward