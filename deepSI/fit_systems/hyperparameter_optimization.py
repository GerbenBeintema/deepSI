import numpy as np
from tqdm.auto import tqdm

def process_dict(search_dict):
    #this is not foul proof so watch out. (e.g. passing lists as items when the whole list should be passed to the function)
    new_dict = {}
    for key,item in search_dict.items():
        if isinstance(item,range):
            new_dict[key] = list(item)
        elif isinstance(item,dict):
            new_dict[key] = process_dict(item)
        elif isinstance(item,(tuple,list,np.ndarray)):
            new_dict[key] = item
        else:
            new_dict[key] = [item]
    return new_dict


def grid_search(fit_system, sys_data, sys_dict_choices={}, fit_dict_choices={}, sim_val=None, RMS=False, verbose=2):
    import copy
    sim_val = sys_data if sim_val is None else sim_val
    #example use: print(grid_search(System_IO_fit_linear,sys_data,sys_dict_choices=dict(na=[1,2,3],nb=[1,2,3]),fit_dict_choices=dict()))

    def itter(sys_dict_choices, fit_dict_choices, sys_dict=None, fit_dict=None, bests=None, best_score=float('inf'), best_sys=None, best_sys_dict=None, best_fit_dict=None):
        if sys_dict is None:
            sys_dict, fit_dict = dict(), dict()
        length_sys_dict, length_fit_dict = len(sys_dict), len(fit_dict)

        if length_sys_dict!=len(sys_dict_choices):
            key,items = sys_dict_choices[length_sys_dict][0], sys_dict_choices[length_sys_dict][1]
            for item in items:
                sys_dict[key] = item
                best_score, best_sys, best_sys_dict, best_fit_dict = itter(sys_dict_choices, fit_dict_choices, sys_dict=sys_dict, fit_dict=fit_dict, best_score=best_score, best_sys=best_sys, best_sys_dict=best_sys_dict, best_fit_dict=best_fit_dict)
            del sys_dict[key]
            return best_score, best_sys, best_sys_dict, best_fit_dict
        elif length_fit_dict!=len(fit_dict_choices):
            key,items = fit_dict_choices[length_fit_dict][0], fit_dict_choices[length_fit_dict][1]
            for item in items:
                fit_dict[key] = item
                best_score, best_sys, best_sys_dict, best_fit_dict = itter(sys_dict_choices, fit_dict_choices, sys_dict=sys_dict, fit_dict=fit_dict, best_score=best_score, best_sys=best_sys, best_sys_dict=best_sys_dict, best_fit_dict=best_fit_dict)
            del fit_dict[key]
            return best_score, best_sys, best_sys_dict, best_fit_dict
        else:

            try: #fit the system if possible
                # print(sys_dict,fit_dict,sys_dict_choices)
                sys = fit_system(**sys_dict)
                sys.fit(sys_data,**fit_dict)
                try:
                    score = sys.apply_experiment(sim_val).RMS(sim_val) if RMS is None else sys.apply_experiment(sim_val).NRMS(sim_val)
                except ValueError:
                    score = float('inf')
            except Exception as inst:
                if verbose>1:
                    print('error',inst,'for sys_dict=', sys_dict, 'for fit_dict=', fit_dict)
                score = float('inf')

            if verbose>1: print(score, sys_dict, fit_dict)
            if score<best_score:
                return score, sys, copy.deepcopy(sys_dict), copy.deepcopy(fit_dict)
            else:
                return best_score, best_sys, best_sys_dict, best_fit_dict

    sys_dict_choices, fit_dict_choices = list(process_dict(sys_dict_choices).items()), list(process_dict(fit_dict_choices).items())
    best_score, best_sys, best_sys_dict, best_fit_dict = itter(sys_dict_choices, fit_dict_choices)
    if verbose>0: print('Result:', best_score, best_sys, best_sys_dict, best_fit_dict)
    return best_score, best_sys, best_sys_dict, best_fit_dict

from random import choice
def sample_dict(dict_now):
    #dict = {'k':[1,2,3],'a':[True,False],}
    #goes to a random choice from the given list
    return dict([(key,sample_dict(item) if isinstance(item,dict) else choice(item)) for key,item in dict_now.items()])

# I can multi process this function in its entirety in the future
def random_search(fit_system, sys_data, sys_dict_choices={}, fit_dict_choices={}, sim_val=None, RMS=False, budget=20,verbose=2):
    sim_val = sys_data if sim_val is None else sim_val
    def test(sys_dict, fit_dict):
        sys = fit_system(**sys_dict)
        try:
            sys.fit(sys_data,**fit_dict)
        except Exception as inst:
            print('error',inst)
            print('for sys_dict=', sys_dict)
            print('for fit_dict=', fit_dict)
        try:
            score = sys.apply_experiment(sim_val).RMS(sim_val) if RMS is None else sys.apply_experiment(sim_val).NRMS(sim_val)
        except ValueError:
            score = float('inf')
        return sys, score

    all_results = []
    sys_dict_choices, fit_dict_choices = process_dict(sys_dict_choices), process_dict(fit_dict_choices)
    try:
        for trial in tqdm(range(budget)):
            sys_dict, fit_dict = sample_dict(sys_dict_choices), sample_dict(fit_dict_choices)
            if verbose>1: print('starting...', sys_dict, fit_dict)
            sys, score = test(sys_dict, fit_dict)
            if verbose>1: print('result:', score ,'for', sys_dict, fit_dict)
            all_results.append(dict(sys_dict=sys_dict,fit_dict=fit_dict,sys=sys,score=score))
    except KeyboardInterrupt:
        print('stopping early due to KeyboardInterrupt')
    return all_results


#GP search may be possible

if __name__ == '__main__':
    import deepSI
    # import numpy as np
    from matplotlib import pyplot as plt
    # import torch
    # from torch import optim, nn
    # from tqdm.auto import tqdm

    from deepSI.datasets.sista_database import destill
    train, test = destill()
    # test.plot()
    # train.plot(show=True)
    sys0 = deepSI.fit_systems.Sklearn_io_linear(na=3,nb=3)
    # sys0.fit(train)
    # sys0.apply_experiment(test).plot()
    # test.plot(show=True)
    choises = dict(na=list(range(6)),nb=list(range(6)))

    best_score, best_sys, best_sys_dict, best_fit_dict = grid_search(deepSI.fit_systems.Sklearn_io_linear, train, sys_dict_choices=choises, fit_dict_choices={}, sim_val=test, verbose=2)

    all_results = random_search(deepSI.fit_systems.Sklearn_io_linear, train, sys_dict_choices=choises, fit_dict_choices={}, sim_val=test, RMS=False, budget=20,verbose=2)


