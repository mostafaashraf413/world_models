#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:25:14 2019

@author: mostafa
"""

import cma
import numpy as np

es = cma.CMAEvolutionStrategy( np.random.rand(3), 0.1, {'popsize': 5})

for it in range(1000):
    solutions = es.ask()
    evaluations = [sum(i)*-1 for i in solutions]
    print(-sum(evaluations))
    es.tell(solutions, evaluations)
    
es.result_pretty()