import numpy as np


class fuzzyDiscretization():

    # Fork note: keeps legacy equifreq/uniform logic and leaves room for empty-split placeholders when features are non-continuous.
    
    def __init__(self, numSet = 7, method='uniform'):
        
        assert method in ['equifreq', 'uniform']
        self.method = method
        assert numSet >=3
        self.numSet = numSet
        
    def run(self, data, continous):
        self.continous = continous
        self.N, self.M = data.shape
        
        splits = []
        for k in range(self.M):
            if self.continous[k]:
                if self.method == 'equifreq':
                    cutPoints = np.sort(data[:,k])[np.linspace(0,self.N-1, self.numSet,  endpoint=True, dtype='int')]

                if self.method == 'uniform':
                    cutPoints = np.linspace(np.min(data[:,k]), np.max(data[:,k]), self.numSet)
                if len(np.unique(cutPoints)) < 3:
                    splits.append(np.array([0.0] + [1.0]*(self.numSet-1)))
                else:
                    uPoints = np.unique(cutPoints)
                    splits.append(np.array(np.append(uPoints, [1.0]*(self.numSet - len(uPoints)))))

            else:
                # new_cPoint = [0.20        , 0., 0.31900178, 0., 1.        ]
                # splits.append(new_cPoint)
                splits.append([])
        
        return splits
