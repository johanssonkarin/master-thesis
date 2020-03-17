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
        self.csv_path = None
        self.dataframe = pd.DataFrame()
        self.isDH = False
        self.isFlex = False
        self.isNew = False
    
    def description(self):
        return ('Residential Load of type {} with ID {}.'.format(self.__class__.__name__, self.ID))
    
    def beFlexible(self):
        if self.isFlex:
            pass
        
    
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
                                     names = ['Date', self.__class__.__name__ + str(self.ID)])
        
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
                                     names = ['Date', self.__class__.__name__ + str(self.ID)])
        
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
                                     names = ['Date', self.__class__.__name__ + str(self.ID)])
        
        
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
                                     names = ['Date', self.__class__.__name__ + str(self.ID)])
    
