from __future__ import print_function

import os.path

import numpy as np
from platypus.algorithms import *
from sklearn.model_selection import train_test_split



# from skmoefs.discretization.discretizer_base import fuzzyDiscretization
from skmoefs.discretization import discretizer_fuzzy as df
from skmoefs.rcs import RCSInitializer, RCSVariator
from skmoefs.toolbox import MPAES_RCS, load_dataset, normalize, is_object_present, store_object, load_object


def set_rng(seed):
    """Set deterministic random seeds."""
    np.random.seed(seed)
    random.seed(seed)

def make_directory(path):
    try:
        os.stat(path)
    except:
        os.makedirs(path)

# def test1():
#     X, y, attributes, inputs, outputs = load_dataset('newthyroid')
#     X_n, y_n = normalize(X, y, attributes)
#     Xtr, Xte, ytr, yte = train_test_split(X_n, y_n, test_size=0.3)

#     my_moefs = MPAES_RCS(capacity=32, variator=RCSVariator(), initializer=RCSInitializer())
#     my_moefs.fit(Xtr, ytr, max_evals=10000)

#     my_moefs.show_pareto()
#     my_moefs.show_pareto(Xte, yte)
#     my_moefs.show_model('median', inputs=inputs, outputs=outputs)

# def test2():
#     X, y, attributes, inputs, outputs = load_dataset('newthyroid')
#     X_n, y_n = normalize(X, y, attributes)
#     my_moefs = MPAES_RCS(variator=RCSVariator(), initializer=RCSInitializer())
#     my_moefs.cross_val_score(X_n, y_n, num_fold=5)


# def test3(dataset, alg, seed, nEvals=50000, store=False):

dataset='iris'
alg='mpaes22'
seed=2
nEvals=200
store=False


path = 'results/' + dataset + '/' + alg + '/'
make_directory(path)
set_rng(seed)
X, y, attributes, inputs, outputs = load_dataset(dataset)
X_n, y_n = normalize(X, y, attributes)

Xtr, Xte, ytr, yte = train_test_split(X_n, y_n, test_size=0.3, random_state=seed)

"""

:param M: Maximum rules per solution (fuzzy classifier).
:param Amin: Minimum antecedents per rule.
:param capacity: Archive capacity.
:param divisions: Grid divisions for the archive.
:param variator: Genetic operator (crossover + mutation).
:param initializer: Defines the initial archive state.
:param objectives: Objectives to optimize (e.g., ('accuracy', 'trl') or ('auc', 'trl')).
:param moea_type: MOEA variant (default MPAES(2+2) if not provided).
"""


Amin = 1
M = 50
capacity = 10
divisions = 8
variator = RCSVariator()
# discretizer = fuzzyDiscretization(numSet=5, method='uniform')
contv = [True] * Xtr.shape[1]
discretizer = df.FuzzyMDLFilter(3, Xtr, ytr, continous=contv)
initializer = RCSInitializer(discretizer=discretizer)
if store:
    base = path + 'moefs_' + str(seed)
    if not is_object_present(base):
        mpaes_rcs_fdt = MPAES_RCS(M=M, Amin=Amin, capacity=capacity,
                                  divisions=divisions, variator=variator,
                                  initializer=initializer, moea_type=alg,
                                  objectives=['accuracy', 'trl'])
        mpaes_rcs_fdt.fit(Xtr, ytr, max_evals=nEvals)
        store_object(mpaes_rcs_fdt, base)
    else:
        mpaes_rcs_fdt = load_object(base)
else:
    mpaes_rcs_fdt = MPAES_RCS(M=M, Amin=Amin, capacity=capacity,
                              divisions=divisions, variator=variator,
                              initializer=initializer, moea_type=alg,
                              objectives=['accuracy', 'trl'])
    mpaes_rcs_fdt.fit(Xtr, ytr, max_evals=nEvals)


complejidad='median'

'''
first-first  = 21 rules   (higher accuracy, lower interpretability)
first-median = 21 rules
first-last   = 21 rules

median-first = 8 rules
median-median= 8 rules
median-last  = 8 rules

last-first   = 4 rules    (lower accuracy, higher interpretability)
last-median  = 4 rules
last-last    = 4 rules
'''

# mpaes_rcs_fdt.show_pareto()
# mpaes_rcs_fdt.show_pareto(Xte, yte)
# mpaes_rcs_fdt.show_pareto_archives()


algo, algo2, algo3, algo4, algo5, algo6, algo7, algo8, algo9 =mpaes_rcs_fdt.show_model(complejidad, inputs, outputs)
# print('CON CAMBIO DE REGLA')
# algo, algo2, algo3, algo4, algo5, algo6, algo7, algo8 =mpaes_rcs_fdt.show_model(complejidad, inputs, outputs)
print('Predicted class:', algo)
print('Rule weights:', algo2)
print('Antecedent matrix:\n', algo3)  # which features and membership functions fired
print('FuzzySet:\n{Feature:MF}', algo8)  # subtract 1 to map to label indices
print('Antecedent dict:', algo4)  # feature indices and their activation
print('Partition matrix:\n', algo5)  # triangle limits per feature
print('Granularity:', algo6)  # number of fuzzy sets per feature
print('Partitions:\n', algo7)  # internal triangle breakpoints
print('Ponderaciones:\n', algo9)


auc, y_pred =mpaes_rcs_fdt.show_predict(complejidad, Xte, yte)

print(y_pred[0],'\n',y_pred[1],'\n',y_pred[2],'\n',y_pred[3])

'''
if (weights[j] * matching_degree) > best_match:
    return y, y_bm, y_md, bm_ind
y: predictions, y_bm: best-match values, y_md: matching degree, bm_ind: index of best-match rule
'''

mpaes_rcs_fdt.show_pareto_archives(Xte, yte)

print(auc)

# md_y=[]
# for i in mpaes_rcs_fdt.classifiers[5].md_app:
#     matching_degree=1
#     for j in i:
#         matching_degree *= j
#     md_y.append(matching_degree)


# for i, w in enumerate(algo2):
#     pred_md = w * y_pred[2][0]
#     print(pred_md)
#     print(y_pred[1][0])
#     print(int(y_pred[3][0]))


# mpaes_rcs_fdt.show_ant()
    
# score,score_archives=mpaes_rcs_fdt.cross_val_score(Xte, yte, nEvals=2000, num_fold=5, seed=2)

# for i in score_archives:
#     print(i)


# if __name__=="__main__":
#     #test1()
#     #test2()
#     test3('iris', 'mpaes22', 2, nEvals=2000, store=False)
