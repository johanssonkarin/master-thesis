# Import packages
from matplotlib import pyplot as plt
from importlib import reload
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import random
import scipy
import math

class ResidentialLoad:
    
    # Initializer
    def __init__(self):
        self.ID = None
        self.csv_path = None
        self.dataframe = pd.DataFrame()
        self.isDH = False
        self.isFlex = False
        self.isNew = False
    
    def description(self):
        return ('Residential Load of type {} with ID {}.'.format(self.__class__.__name__, self.ID))
    
    def be_flexible(self, date_list, reduction):
        self.isFlex = True
        self.flexSlack = random.randint(4,7) #hours to freeze recovery, 4-6
        self.flexRecover = random.randint(3,9) #hours to recover consumption, 3-8
        self.dataframe.fillna(0, inplace = True) #move this to better spot
        self.dataframe.index.map(lambda x: self.flex(x,date_list, reduction))


        
    def flex(self, index, date_list, reduction): #implement moving comsumption to both before and after
        if index.normalize() in date_list and index.hour in [17,18,19]: 
            reduce = reduction * self.dataframe.loc[index].item() #from percent to value
            self.dataframe.loc[index] -= reduce #reduce from peak
            loc = self.dataframe.index.get_loc(index) 
            i_start = - int(index.hour) + 19 + self.flexSlack #when to start adding
            i_end = i_start + self.flexRecover #when to stop adding
            reduce /= i_end #calculate hourly increase
            self.dataframe.iloc[loc+i_start:loc+i_end] += reduce #increse

            
    
class HouseNew(ResidentialLoad):
    # Initializer
    def __init__(self, ID):
        self.ID = ID
        self.csv_path = '../data/new_houses.csv'
        self.isDH = False
        self.isFlex = False
        self.isNew = True
        self.dataframe = pd.read_csv(self.csv_path, 
                                     index_col = 0, 
                                     parse_dates = True, 
                                     usecols = [0,random.randrange(1,49,1)],
                                     skiprows = 1,
                                     names = ['Date', self.ID])
        # previous column name including load type:
        # self.__class__.__name__ + str(self.ID)
        
class HouseOld(ResidentialLoad):
    # Initializer
    def __init__(self, ID):
        self.ID = ID
        self.csv_path = '../data/old_houses.csv'
        self.isDH = False
        self.isFlex = False
        self.isNew = False
        self.dataframe = pd.read_csv(self.csv_path, 
                                     index_col = 0, 
                                     parse_dates = True,
                                     usecols = [0, random.randrange(1,35,1)],
                                     skiprows = 1,
                                     names = ['Date', self.ID])

        
class HouseDH(ResidentialLoad):
    # Initializer
    def __init__(self, ID):
        self.ID = ID
        self.csv_path = '../data/mixed_ages_houses_district_heating.csv'
        self.isDH = True
        self.isFlex = False
        self.isNew = False
        self.dataframe = pd.read_csv(self.csv_path, 
                                     index_col = 0, 
                                     parse_dates = True, 
                                     usecols = [0,random.randrange(1,37,1)],
                                     skiprows = 1,
                                     names = ['Date', self.ID])
        
        
class ApartmentNewDH(ResidentialLoad):
    # Initializer
    def __init__(self, ID):
        self.ID = ID
        self.csv_path = '../data/new_apartments_district_heating.csv'
        self.isDH = False
        self.isFlex = False
        self.isNew = True
        self.dataframe = pd.read_csv(self.csv_path, 
                                     index_col = 0, 
                                     parse_dates = True, 
                                     usecols = [0, random.randrange(1,35,1)],
                                     skiprows = 1,
                                     names = ['Date', self.ID])
    
