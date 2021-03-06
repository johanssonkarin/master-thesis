from residentialload import HouseNew, HouseOld, HouseDH, ApartmentNewDH
from office import Office
from PV import PV
from substation import Substation

from matplotlib import pyplot as plt
from importlib import reload
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import random
import scipy
import math

class Grid:
    '''
    Grid class.
    This class represents a power grid.

    Attributes
    ----------
    ID_count: int
        To keep track of station IDs.
    station_count: int
        Number of substations in the net.
    station_dict: dict
        Contains all substation objects.
        IDs are keys.
    '''

    def __init__(self):
        self.ID_count = 0
        self.station_count = 0
        self.station_dict = dict()
        self.dataframe = pd.DataFrame()


    def add_station(self, station):
        '''Adds a substation to the net.'''
        self.ID_count += 1
        self.station_count += 1
        self.station_dict[self.ID_count] = station
        
        if not 'AggregatedLoad' in station.dataframe.columns:
            station.update_aggregated_col()
            
        if self.dataframe.empty:
                        self.dataframe = pd.DataFrame(index=station.dataframe.index)
                        self.dataframe[self.ID_count]= station.dataframe['AggregatedLoad']
                        
        else:
            self.dataframe[self.ID_count] = station.dataframe.loc[station.dataframe.index.isin(self.dataframe.index),['AggregatedLoad']]
            
        self.update_aggregated_col()


    def del_station(self, key):
        '''Deletes a substation from the grid using ID.'''
        self.station_count -= 1
        del self.station_dict[key]
        self.dataframe.drop(columns = key,
                            inplace = True)
        self.update_aggregated_col()
        

    def description(self):
        '''Returns minimal description of the grid.'''
        if self.station_count == 0:
            return 'An empty power grid.'
        if self.station_count == 1:
            return 'A net with 1 substation.'
        return 'A net with {} substations'.format(self.station_count)


    def calculate_norm(self):
        '''Calculates mean and standard deviation.'''
        self.update_aggregated_col()
        mu, sigma = scipy.stats.norm.fit(self.dataframe['AggregatedLoad'].tolist())
        self.mu, self.sigma = round(mu,3), round(sigma,3)
        
        

    def create_date_cols(self):
        ''' Auxillary method to create extra datetime info columns.'''
        self.dataframe['Year'] = self.dataframe.index.year
        self.dataframe['Month'] = self.dataframe.index.month
        self.dataframe['Weekday'] = self.dataframe.index.weekday_name
        self.dataframe['Hour'] = self.dataframe.index.hour



    def find_max(self):
        '''
        To find the maximal hourly consumtion
        of the grid object.

        Returns
        -------
        Timestamp. 
        '''
        
        if not 'AggregatedLoad' in self.dataframe.columns:
            self.update_aggregated_col()
        return self.dataframe['AggregatedLoad'].idxmax(axis = 0)
        


    def update_aggregated_col(self):
        '''
        Updates the column 'AggregatedLoad' representing
        the sum of all consumption at that timestamp.
        If the column doesn't exist, it is added.
        
        '''
        self.dataframe.sort_index(inplace=True) # making sure df is sorted
        self.dataframe['AggregatedLoad'] = self.dataframe.loc[:,self.dataframe.columns.isin(range(1,self.ID_count+1))].sum(numeric_only=True, axis=1) # update sum col
