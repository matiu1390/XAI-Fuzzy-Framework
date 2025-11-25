# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:29:00 2024

@author: Lab_iHealth
"""


import numpy as np

# partitions=[np.array([0.        , 0.28400228, 0.52962495, 0.91489319]), np.array([0.        , 0.18669341, 0.60363678, 1.        ]), np.array([0.        , 0.26063513, 0.51550162, 1.        ]), np.array([0.        , 0.26516452, 0.82235703, 1.        ]), np.array([0.        , 0.79411866, 0.86274577, 0.99019319])]

partitions=[np.array([0.        , 0.34269456, 0.50332616, 0.91489319]), np.array([0.02616951, 0.17234763, 0.51014744, 1.        ]), np.array([0.        , 0.30870509, 0.49320449, 1.        ]), np.array([0.        , 0.15617794, 0.73393868, 1.        ]), np.array([0.        , 0.79658297, 0.82049507, 1.        ])]

attributes= [[61.66667, 140.0], [16.609, 43.3959], [18.0, 45.2], [1.0, 6.0], [27.8571, 42.4286]]

f= open("rule_case.txt", 'w')

for k, partition in enumerate(partitions):
    
    anchors = np.concatenate(([partition[0]], partition, [partition[-1]]))
    fuzzyset_size= len(anchors) - 2
    
    for i in range(fuzzyset_size):
        anchors_temp=anchors.copy()    
        anchors_temp[i:i+3]=anchors[i:i+3]*(attributes[k][1]-attributes[k][0])+attributes[k][0]
        triangle=anchors_temp[i:i+3]
        f.write(str(triangle[1])+' '+str(triangle[1])+' '+ 
                str(triangle[1]-triangle[0])+' '+ str(triangle[2]-triangle[1])+ '\n')
        print(triangle)