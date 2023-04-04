# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:11:58 2022

@author: emili
"""
import numpy as np
import pandas as pd
from decimal import Decimal

data = np.genfromtxt('Muscovite_ninduced_wan.dat',
                     skip_header=2,
                     names=True,
                     dtype=None,
                     delimiter='  ')

print(data)
list_col = list(range(0,13))
df = pd.DataFrame(data = data)


df = df.drop('2284174E06', axis=1)
df = df.drop('5969682E06', axis=1)
df = df.drop('3503334E05', axis=1)
df = df.drop('6773113E05', axis=1)
#df = df.apply(pd.to_numeric, errors='coerce')

#df.to_csv('text file from csv musc ninduced.txt', sep='\t', index=False)
