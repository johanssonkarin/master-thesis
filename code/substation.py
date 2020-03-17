from residentialload import HouseNew, HouseOld, HouseDH, ApartmentNewDH
from matplotlib import pyplot as plt
from importlib import reload
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import random
import scipy
import math

class Substation:

    
    # Class Attributes
    region_path_dict = {'Stockholm': '../data/stockholm_mintemp.csv'}
    
    # Initializer / Instance Attributes
    def __init__(self, region):
        self.load_dict = dict()
        self.dataframe = pd.DataFrame()
        self.load_count = 0
        self.house_count = 0
        self.apartment_count = 0
        self.flex_count = 0
        self.DH_count = 0
        self.region = region
        self.isFlex = False
        self.mu = None
        self.sigma = None
        self.start = None
        self.end = None
        
    def add_residential_load(self, load_type, num = 1):
        if num > 0:
            if load_type == 'HouseNew':
                for i in range(0,num):
                    self.load_count += 1
                    self.house_count += 1
                    load = HouseNew(ID = self.load_count)
                    self.load_dict[self.load_count] = load 
                    
                    if self.dataframe.empty:
                        self.dataframe = load.dataframe
                    else:
                        self.dataframe = self.dataframe.merge(load.dataframe,
                                                              how = 'inner',
                                                              left_index=True,
                                                              right_index=True)
            elif load_type == 'HouseOld':
                for i in range(0,num):
                    self.load_count += 1
                    self.house_count += 1
                    load = HouseOld(ID = self.load_count)
                    self.load_dict[self.load_count] = load 
                    
                    if self.dataframe.empty:
                        self.dataframe = load.dataframe
                    else:
                        self.dataframe = self.dataframe.merge(load.dataframe,
                                                              how = 'inner',
                                                              left_index=True,
                                                              right_index=True)
            elif load_type == 'HouseDH':
                for i in range(0,num):
                    self.load_count += 1
                    self.house_count += 1
                    self.DH_count += 1
                    load = HouseDH(ID = self.load_count)
                    self.load_dict[self.load_count] = load 
                    
                    if self.dataframe.empty:
                        self.dataframe = load.dataframe
                    else:
                        self.dataframe = self.dataframe.merge(load.dataframe,
                                                              how = 'inner',
                                                              left_index=True,
                                                              right_index=True)
            elif load_type == 'ApartmentNewDH':
                for i in range(0,num):
                    self.load_count += 1
                    self.apartment_count += 1
                    self.DH_count += 1
                    load = ApartmentNewDH(ID = self.load_count)
                    self.load_dict[self.load_count] = load 
                    
                    if self.dataframe.empty:
                        self.dataframe = load.dataframe
                    else:
                        self.dataframe = self.dataframe.merge(load.dataframe,
                                                              how = 'inner',
                                                              left_index=True,
                                                              right_index=True)
                        
            self.update_dates(self.dataframe.index[0],self.dataframe.index[-1])
        
        
    def update_dates(self, start, end):
        self.start = start
        self.end = end
        
    def calculate_norm(self):
        if 'AggregatedLoad' not in self.dataframe.columns:
            self.add_aggregated_col()
        mu, sigma = scipy.stats.norm.fit(self.dataframe['AggregatedLoad'].tolist())
        self.mu, self.sigma = round(mu,3), round(sigma,3)
        
    def create_date_cols(self):
        self.dataframe['Year'] = self.dataframe.index.year
        self.dataframe['Month'] = self.dataframe.index.month
        self.dataframe['Weekday'] = self.dataframe.index.weekday_name
        self.dataframe['Hour'] = self.dataframe.index.hour
        
    # Function which takes a dataframe where
    # each column represents a load and rows = date/time
    # returns same dataframe but with a aggregated column.
    def add_aggregated_col(self):
        self.dataframe.sort_index(inplace=True) # making sure df is sorted
        self.dataframe['AggregatedLoad'] = self.dataframe.sum(numeric_only=True, axis=1) # add new sum col

        
    def description(self):
        if self.mu == None:
            self.calculate_norm()
        return ('Substation based on data from {} to {}.'.format(self.start,self.end)\
               + ' The substation contains {} loads with an '.format(self.load_count)\
                + 'aggregated average comsumption of {} (-/+ {}) kWh per hour.'.format(self.mu,self.sigma))

    # Function cutting dataframe to whole years.
    # By default jan-dec but can be changed to whole
    # years from first date index. 
    def filter_whole_years(self, jan_start = False):
        first_date, last_date = self.start, self.end

        if jan_start:
            start_date, end_date = str(first_date.year+1) +'-01-01', str(last_date.year-1) +'-12-31'
        else:
            max_years = math.floor((last_date - first_date) / datetime.timedelta(days=365))
            start_date = str(first_date).split(' ',1)[0]
            end_date = str(first_date + datetime.timedelta(days = (365 * max_years))).split(' ',1)[0]

        self.dataframe = self.dataframe[start_date:end_date]
        self.update_dates(self.dataframe.index[0],self.dataframe.index[-1])

        
    # Function that takes a sorted list of load demand 
    # values and produces a plot of the load duration curve.
    def plot_load_duration_curve(self,sorted_demand_list):
        list_len = len(sorted_demand_list) #Number of datapoints
        x = np.linspace(1,list_len,list_len).tolist() #List of hours

        plt.plot(x,sorted_demand_list)
        plt.title('Load Duration curve')
        plt.xlabel('Hours')
        plt.ylabel('Consumption [kWh]') #Review if kwh or not later on
        plt.show()
    
    # Function for generating and printing different kinds of
    # information about the dataframe and load profiles. 
    def print_insights(self, 
                       duration_curve = True,
                       month_plot = True, 
                       weekday_plot = True, 
                       hour_plot = True):
        
        if 'AggregatedLoad' not in self.dataframe.columns:
            self.add_aggregated_col()
        if 'Month' not in self.dataframe.columns: 
            self.create_date_cols()
            
        if duration_curve:
            col_lst = self.dataframe['AggregatedLoad'].sort_values().tolist()
            self.plot_load_duration_curve(col_lst)
    
        if month_plot:
           # 'exec(%matplotlib inline)'
            reload(plt)
            'exec(%matplotlib notebook)'
            ax = sns.boxplot(data=self.dataframe, x='Month', y='AggregatedLoad')
            ax.set_ylabel('kWh')
            ax.set_title('Hourly comsumption of the substation')
        
        if weekday_plot:
           # %matplotlib inline
            reload(plt)
            'exec(%matplotlib notebook)'
            sns.set(style="whitegrid")
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            ax1 = sns.boxplot(data=self.dataframe, x='Weekday', y='AggregatedLoad', order=order)
            ax1.set_ylabel('kWh')
            ax1.set_xlabel('')
            ax1.set_title('Hourly comsumption of the substation')
            
        if hour_plot:
           # %matplotlib inline
            reload(plt)
            'exec(%matplotlib notebook)'
            ax = sns.boxplot(data=self.dataframe, x='Hour', y='AggregatedLoad')
            ax.set_ylabel('kWh')
            ax.set_title('Hourly comsumption of the substation')            
            
