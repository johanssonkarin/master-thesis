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
    '''
    Class representing residential loads.

    Attributes
    -----------
    ID: int
        Should be unique.
    csv_path: str
        Indicates where the consumption data is
        drawn from.
    is_DH: bool
        If the heating source of the
        load is District Heating or not.
    is_flex: bool
        If the load implements static flex
        or not.
    is_new: bool
        If the load is a new or old building.
        If not known, it is labeled old, i.e.
        False.
    
    '''
    
    # Initializer
    def __init__(self):
        self.ID = None
        self.csv_path = None
        self.dataframe = pd.DataFrame()
        self.is_DH = False
        self.is_flex = False
        self.is_new = False
    
    def description(self):
        '''
        Method generating a minimal description of
        the residential load objects.

        Returns
        -------
        String.

        '''
        return ('Residential Load of type {} with ID {}.'.format(self.__class__.__name__, self.ID))
    
    def be_flexible(self, date_list, reduction, move_both = True):
        '''
        Method which changes the consumption values within
        the dataframe attribute according to the static flex
        implementation.

        Parameters
        ----------
        date_list: list
            A list of the dates to reduce consumption.
        reduction: float
            Indicates a percentage to be reduced from the values.
        move_both: bool, optional
            If the reduced consumption should be move both
            forward or backwards or if just forward.

        Returns
        -------
        None.
        
        '''
        self.is_flex = True
        self.flexSlack = random.randint(5,7) #hours to freeze recovery, 5-7
        self.flexRecover = random.randint(2,5) #hours to recover consumption, 2-4
        self.dataframe.fillna(0, inplace = True) #move this to better spot maybe
        self.dataframe.index.map(lambda x: self.flex(x,date_list, reduction, move_both = move_both))


        
    def flex(self, index, date_list, reduction, move_both, hour_list = [17,18,19]):
        '''
        Auxillary method for the be_flexible method. THe actual reduction and
        redistribution happen within this method.

        Parameters
        ----------
        index: datetimeindex
            All indices of the dataframe.
        date_list: list
            The list of dates to reduce.
        reduction: float
            The consumption reduction.
        move_both: bool
            How to redistribute the consumption.
            Only forward or forward and backwards.
        hour_list: list, optional
            Whcich hours to reduce consumption.
            By default the peak hours. 
        

        Returns
        -------
        None.

        '''
        if index.normalize() in date_list and index.hour in hour_list:
            reduce = reduction * self.dataframe.loc[index].values #from percent to value
            self.dataframe.loc[index] -= reduce #reduce from peak
            loc = self.dataframe.index.get_loc(index) 
            if move_both: #forward and backwards
                # i is before
                i_end = - int(index.hour) + 17 - int(self.flexSlack/2) #when to start adding
                i_start = i_end - int(self.flexRecover/2) #when to stop adding
                # j is after
                j_start = - int(index.hour) + 19 + int(self.flexSlack/2) #when to start adding
                j_end = j_start + int(self.flexRecover/2) #when to stop adding
                reduce /= j_end * 2 #calculate hourly increase
                self.dataframe.iloc[loc+i_start:loc+i_end] += reduce #increse before
                self.dataframe.iloc[loc+j_start:loc+j_end] += reduce #increse after
            else: #just forward
                j_start = - int(index.hour) + 19 + self.flexSlack #when to start adding
                j_end = j_start + self.flexRecover #when to stop adding
                reduce /= j_end #calculate hourly increase
                self.dataframe.iloc[loc+j_start:loc+j_end] += reduce #increse

            
    
class HouseNew(ResidentialLoad):
    # Initializer subclass
    def __init__(self, region, ID):
        self.ID = ID
        self.csv_path = '../data/'+region+'/Residential/new_houses.csv'
        self.is_DH = False
        self.is_flex = False
        self.is_new = True
        self.dataframe = pd.read_csv(self.csv_path, 
                                     index_col = 0, 
                                     parse_dates = True, 
                                     usecols = [0,random.randrange(1,49,1)],
                                     skiprows = 1,
                                     names = ['Date', self.ID])
        # previous column name including load type:
        # self.__class__.__name__ + str(self.ID)
        
class HouseOld(ResidentialLoad):
    # Initializer subclass
    def __init__(self, region, ID):
        self.ID = ID
        self.csv_path = '../data/'+region+'/Residential/old_houses.csv'
        self.is_DH = False
        self.is_flex = False
        self.is_new = False
        self.dataframe = pd.read_csv(self.csv_path, 
                                     index_col = 0, 
                                     parse_dates = True,
                                     usecols = [0, random.randrange(1,35,1)],
                                     skiprows = 1,
                                     names = ['Date', self.ID])

        
class HouseDH(ResidentialLoad):
    # Initializer subclass
    def __init__(self, region, ID):
        self.ID = ID
        self.csv_path = '../data/'+region+'/Residential/mixed_ages_houses_district_heating.csv'
        self.is_DH = True
        self.is_flex = False
        self.is_new = False
        self.dataframe = pd.read_csv(self.csv_path, 
                                     index_col = 0, 
                                     parse_dates = True, 
                                     usecols = [0,random.randrange(1,37,1)],
                                     skiprows = 1,
                                     names = ['Date', self.ID])
        
        
class ApartmentNewDH(ResidentialLoad):
    # Initializer subclass
    def __init__(self, region, ID):
        self.ID = ID
        self.csv_path = '../data/'+region+'/Residential/new_apartments_district_heating.csv'
        self.is_DH = False
        self.is_flex = False
        self.is_new = True
        self.dataframe = pd.read_csv(self.csv_path, 
                                     index_col = 0, 
                                     parse_dates = True, 
                                     usecols = [0, random.randrange(1,35,1)],
                                     skiprows = 1,
                                     names = ['Date', self.ID])
    
