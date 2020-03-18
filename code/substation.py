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
        self.coldest_days = []
        
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
            self.update_aggregated_col()
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
    def update_aggregated_col(self):
        self.dataframe.sort_index(inplace=True) # making sure df is sorted
        self.dataframe['AggregatedLoad'] = self.dataframe.loc[:,range(1,self.load_count+1)].sum(numeric_only=True, axis=1) # update sum col

        
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
        
        self.update_aggregated_col()
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

    ### ----- Flex-related ---------------------------------------
    
    def introduce_flexibility(self, 
                              days = 17, 
                              percent_loads = 0.5, 
                              reduction = 0.65, 
                              onlyDH = True):
        '''
        Dummy version for adding demand side flexibility trend to 
        the substation. 'num' is the number of flexible days
        per year which is then translated into from the coldest days
        The model assumes a percentage of loads (default 50%) can be
        flexible, which can be DH or not. Flexible is interpreted as 
        reducing comsumption with a certain percentage (default 0.67) 
        and then redistributedaccording to the slack parameter which 
        is defined in hours.
        '''
        self.isFlex = True
        if not self.coldest_days:
            self.find_coldest_days(days)
        
        if onlyDH:
            ID_list = self.get_loadID(noDH = True)
        else:
            ID_list = self.dataframe.columns #get all IDs

        ID_list = self.remove_percentage_list(ID_list, percent_loads)
        for ID in ID_list:
            self.flex_count += 1
            self.load_dict[ID].be_flexible(self.coldest_days, reduction)
            #self.dataframe.loc[:,ID] = self.load_dict[ID].dataframe
            self.dataframe.loc[self.dataframe.index.isin(self.load_dict[ID].dataframe.index), self.load_dict[ID].dataframe.columns] = self.load_dict[ID].dataframe.loc[self.load_dict[ID].dataframe.index.isin(self.dataframe.index), self.load_dict[ID].dataframe.columns].values

        

    def remove_percentage_list(self, thelist, percentage):
        random.shuffle(thelist)
        count = int(len(thelist) * percentage)
        if not count: return []  # edge case, no elements removed
        return thelist[-count:]        
        
    
    def find_coldest_days(self, num):
        '''
        Function for finding the n (default same as number of flexible 
        days: 17) coldest days per year in a region, which is then updated 
        in the substation attribute 'coldest_days'. The path to the 
        temperature data needs to be specified within the class
        attribute 'region_path_dict'. The start and end of the timeframe to
        check is determined by the start and end attributes of the object. 
        '''
        temp_data = pd.read_csv(self.region_path_dict[self.region], index_col = 0, parse_dates = True)

        temp_data = temp_data[self.start:self.end]
        temp_data['Year'] = temp_data.index.year
        num_years = temp_data['Year'].nunique()
        list_of_dates = []

        for i in range(num_years):
            year = self.start.year + i
            year_data = temp_data.loc[temp_data['Year'] == year]
            list_of_dates += year_data.nsmallest(num, columns='Temperature').index.sort_values().tolist()

        self.coldest_days = list_of_dates

        
    def get_loadID(self, 
                   noDH = False, 
                   flex = False):
        '''
        Function that returns ID of loads of the substation
        based on attribute.
        '''
        if noDH:
            ID_list = [ID for ID,obj in self.load_dict.items() if not obj.isDH]
        if flex:
            ID_list = [ID for ID,obj in self.load_dict.items() if obj.isFlex]
        return ID_list
            
