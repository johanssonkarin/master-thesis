from residentialload import HouseNew, HouseOld, HouseDH, ApartmentNewDH
from office import Office
from PV import PV

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
    '''
    Substation class.
    This class represents an electrical substation.

    Parameters
    ----------
    region : str
        Location of the substation, e.g. 'Stockholm',
        needs to match data folder name.    
    
    
    Attributes
    ----------
    load_dict: dict
        Contains all residential load objects.
        IDs are keys.
    pv_dict: dict
        Contains all PV objects.
        IDs are keys.
    office_dict: dict
        Contains all PV objects.
        IDs are keys.
    dataframe: pandas DataFrame
        Every object is a column. DateTimeIndex.
    ID_count: int
        Keeps track of IDs.
    load_count: int
        Number of residential loads.
    house_count: int
        Number of residential houses.
    apartment_count: int
        Number of appartments.
    office_count: int
        Number of offices.
    flex_count:
        Number of flexible agents.
    DH_count: int
        Number of loads with district heatning.
    PV_count: int
        Number of PV plants. 
    region: str
        Where the substation is located,
    is_flex: bool
        If there are flexible loads in the substation.
    is_efficient: bool
        If the substation makes loads more efficent.
    mu: float
        The average hourly consumption of the
        substation.
    sigma: float
        Standard deviation of hourly cosumption of
        the substation.
    start: str
        Start date of the time series data.
    end: 
        End date of the time series data.
    
    '''
    
    
    # Initializer / Instance Attributes
    def __init__(self, region):
        self.load_dict = dict()
        self.pv_dict = dict()
        self.office_dict = dict()
        self.dataframe = pd.DataFrame()
        self.ID_count = 0
        self.load_count = 0
        self.house_count = 0
        self.apartment_count = 0
        self.office_count = 0
        self.flex_count = 0
        self.DH_count = 0
        self.PV_count = 0
        self.region = region
        self.is_flex = False
        self.is_efficient = False
        self.mu = None
        self.sigma = None
        self.start = None
        self.end = None
        self.coldest_days = []
        self.region_path = '../data/'+region+'/min_temp.csv'



    ###--------------- ADD LOADS ------------------------------
        
    def add_residential_load(self, load_type, num = 1):
        if num > 0:
            if load_type == 'HouseNew':
                for i in range(0,num):
                    self.ID_count += 1
                    self.load_count += 1
                    self.house_count += 1
                    load = HouseNew(region = self.region, ID = self.ID_count)
                    self.load_dict[self.ID_count] = load 
                    
                    if self.dataframe.empty:
                        self.dataframe = load.dataframe
                    else:
                        self.dataframe = self.dataframe.merge(load.dataframe,
                                                              how = 'inner',
                                                              left_index=True,
                                                              right_index=True)
            elif load_type == 'HouseOld':
                for i in range(0,num):
                    self.ID_count += 1
                    self.load_count += 1
                    self.house_count += 1
                    load = HouseOld(region = self.region, ID = self.ID_count)
                    self.load_dict[self.ID_count] = load 
                    
                    if self.dataframe.empty:
                        self.dataframe = load.dataframe
                    else:
                        self.dataframe = self.dataframe.merge(load.dataframe,
                                                              how = 'inner',
                                                              left_index=True,
                                                              right_index=True)
            elif load_type == 'HouseDH':
                for i in range(0,num):
                    self.ID_count += 1
                    self.load_count += 1
                    self.house_count += 1
                    self.DH_count += 1
                    load = HouseDH(region = self.region, ID = self.ID_count)
                    self.load_dict[self.ID_count] = load 
                    
                    if self.dataframe.empty:
                        self.dataframe = load.dataframe
                    else:
                        self.dataframe = self.dataframe.merge(load.dataframe,
                                                              how = 'inner',
                                                              left_index=True,
                                                              right_index=True)
            elif load_type == 'ApartmentNewDH':
                for i in range(0,num):
                    self.ID_count += 1
                    self.load_count += 1
                    self.apartment_count += 1
                    self.DH_count += 1
                    load = ApartmentNewDH(region = self.region, ID = self.ID_count)
                    self.load_dict[self.ID_count] = load 
                    
                    if self.dataframe.empty:
                        self.dataframe = load.dataframe
                    else:
                        self.dataframe = self.dataframe.merge(load.dataframe,
                                                              how = 'inner',
                                                              left_index=True,
                                                              right_index=True)
                        
            self.update_dates(self.dataframe.index[0],self.dataframe.index[-1])

            
    def add_office(self, num = 1, randomize = True):
        '''
        Stupid function for addinf office like load curves
        to the substation.By default the offices are randomized
        by a percentage drawn from a gaussion distribution
        with mu = 0 and sigma = 0.1.
        '''
        for i in range(0,num):
            self.ID_count += 1
            self.office_count +=1

            office = Office(start = self.start,
                            end = self.end,
                            ID = self.ID_count)

            # Save office object to dict
            self.office_dict[office.ID] = office
            
            if randomize:
                office.be_random()

            # Add office to substation dataframe
            self.dataframe = self.dataframe.merge(office.dataframe, 
                                                  how = 'inner',
                                                  left_index=True,
                                                  right_index=True)
        

    def add_PV(self, size, num = 1, randomize = True):
        '''
        Function for adding PV plants to the substation.
        By default these are randomized by a percentage drawn
        from a gaussion distribution (mu = 0, sigma = 0.1).
        '''
        for i in range(0,num):
            self.ID_count += 1
            self.PV_count += 1
            
            pv = PV(size = size,
                    region = self.region,
                    start = str(self.start).split(' ',1)[0],
                    end = str(self.end).split(' ',1)[0],
                    ID = self.ID_count)
            
            if randomize:
                pv.be_random()

            # Save PV object to dict
            self.pv_dict[pv.ID] = pv
            
            # PV production means neagtive consumption. Merge negative values
            pv_df_neg = pv.dataframe
            pv_df_neg[pv.ID] *= (-1)
            self.dataframe = self.dataframe.merge(pv_df_neg,
                                                  how = 'inner',
                                                  left_index=True,
                                                  right_index=True)

    ### ----------- SUBSTATION RELATED -------------------------------
        
    def update_dates(self, start, end):
        self.start = start
        self.end = end
        
    def calculate_norm(self):
        self.update_aggregated_col()
        mu, sigma = scipy.stats.norm.fit(self.dataframe['AggregatedLoad'].tolist())
        self.mu, self.sigma = round(mu,3), round(sigma,3)
        
        
    def create_date_cols(self):
        self.dataframe['Year'] = self.dataframe.index.year
        self.dataframe['Month'] = self.dataframe.index.month
        self.dataframe['Weekday'] = self.dataframe.index.weekday_name
        self.dataframe['Hour'] = self.dataframe.index.hour


    def find_max(self):
        '''
        Returns date and time of largest
        hourly consumtion of the substation.

        Returns
        -------
        Timestamp. 
        '''
        
        if 'AggregatedLoad' not in self.dataframe.columns:
            self.update_aggregated_col()
        return self.dataframe['AggregatedLoad'].idxmax(axis = 0)
        

        

    def update_aggregated_col(self):
        '''
        Updates the column 'AggregatedLoad' representing
        the sum of all consumption at that timestamp.
        
        '''
        self.dataframe.sort_index(inplace=True) # making sure df is sorted
        self.dataframe['AggregatedLoad'] = self.dataframe.loc[:,self.dataframe.columns.isin(range(1,self.ID_count+1))].sum(numeric_only=True, axis=1) # update sum col



    def copy_load_stochastic(self, column_name, sigma=0.1, inplace = False):
        '''
        A function to copy a load profile with stochastic deviation
        from the original. The values varies according to a gaussian 
        distribution, with default and mu = 0, sigma = 0.1. The
        funtion returns a copy of the dataframe with the new column.
        '''
        min_prob, max_prob = -sigma, sigma
        prob_array = (max_prob - min_prob) * np.random.random_sample(size=self.dataframe.shape[0]) + min_prob
        
        if inplace:
            self.dataframe[column_name] += self.dataframe[column_name].mul(prob_array)
        else:
            new_col_name = str(column_name) + '_stoch_copy'
            self.dataframe[new_col_name] = self.dataframe[column_name] + self.dataframe[column_name].mul(prob_array)
        


        
    def description(self):
        '''Returns a description of the substation object.'''
        if self.mu == None:
            self.calculate_norm()
        return ('Substation based on data from {} to {}.'.format(self.start,self.end)\
               + ' The substation contains {} loads with an '.format(self.load_count)\
                + 'aggregated average comsumption of {} (-/+ {}) kWh per hour.'.format(self.mu,self.sigma))
    


    

    def filter_whole_years(self, jan_start = False, num = 0):
        '''
        Function cutting dataframe to whole years.
        By default jan-dec but can be changed to whole
        years from first date index. The 'num'
        parameter specifies number of years to keep
        and needs to minimum 1. If num is not specified,
        maximum number of years are kept. 
        '''
        first_date, last_date = self.start, self.end

        if num >= 1:
            if jan_start:
                if first_date.month in [1] and first_date.day in [1]:
                    start_date = str(first_date.year) +'-01-01'
                    end_date = str(first_date.year+num-1) +'-12-31'
                else:
                    start_date = str(first_date.year+1) +'-01-01'
                    end_date = str(first_date.year+num) +'-12-31'
            else:
                start_date = str(first_date).split(' ',1)[0]
                end_date = str(first_date + datetime.timedelta(days = (365 * num))).split(' ',1)[0]
                
        else:
            if jan_start:
                if first_date.month in [1] and first_date.day in [1]:
                    start_date = str(first_date.year) +'-01-01'
                    end_date = str(last_date.year-1) +'-12-31'
                else:
                    start_date = str(first_date.year+1) +'-01-01'
                    end_date = str(last_date.year-1) +'-12-31' 
                
            else:
                max_years = math.floor((last_date - first_date) / datetime.timedelta(days=365))
                start_date = str(first_date).split(' ',1)[0]
                end_date = str(first_date + datetime.timedelta(days = (365 * max_years))).split(' ',1)[0]

        self.dataframe = self.dataframe[start_date:end_date]
        self.update_dates(self.dataframe.index[0],self.dataframe.index[-1])


        

    def plot_load_duration_curve(self,sorted_demand_list):
        '''
        Function that takes a sorted list of load demand 
        values and produces a plot of the load duration curve.
        '''
        list_len = len(sorted_demand_list) # Number of datapoints
        x = np.linspace(1,list_len,list_len).tolist() # List of hours

        plt.plot(x,sorted_demand_list)
        plt.title('Load Duration curve')
        plt.xlabel('Hours')
        plt.ylabel('Consumption [kWh]') #Review if kwh or not later on
        plt.show()
    



    def print_insights(self, 
                       duration_curve = True,
                       month_plot = True, 
                       weekday_plot = True, 
                       hour_plot = True):
        '''
        Function for generating and printing different kinds of
        information about the dataframe and load profiles. 
        '''
        
        self.update_aggregated_col()
        if 'Month' not in self.dataframe.columns: 
            self.create_date_cols()
            
        if duration_curve:
            col_lst = self.dataframe['AggregatedLoad'].sort_values(ascending=False).tolist()
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


            

    ### ----- FLEX RELATED ---------------------------------------
    
    def introduce_flexibility(self, 
                              days = 17, 
                              percent_loads = 0.5, 
                              reduction = 0.65, 
                              only_noDH = True):
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
        self.is_flex = True
        if not self.coldest_days:
            self.find_coldest_days(days)
        
        if only_noDH:
            ID_list = self.get_loadID(noDH = True)
        else:
            ID_list = self.get_loadID(residential = True) #get all IDs

        ID_list = self.remove_list_element(ID_list, percentage = percent_loads)
        for ID in ID_list:
            self.flex_count += 1
            self.load_dict[ID].be_flexible(self.coldest_days, reduction)
            #self.dataframe.loc[:,ID] = self.load_dict[ID].dataframe
            self.dataframe.loc[self.dataframe.index.isin(self.load_dict[ID].dataframe.index), self.load_dict[ID].dataframe.columns] = self.load_dict[ID].dataframe.loc[self.load_dict[ID].dataframe.index.isin(self.dataframe.index), self.load_dict[ID].dataframe.columns].values

        

    def remove_list_element(self, thelist, percentage = None, num = None):
        random.shuffle(thelist)
        if percentage:
            count = int(len(thelist) * percentage)
        elif num:
            count = num
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
        temp_data = pd.read_csv(self.region_path, index_col = 0, parse_dates = True)

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
                   flex = False,
                   residential = False,
                   office = False):
        '''
        Function that returns ID of loads of the substation
        based on attribute.
        '''
        ID_list = []
        
        if noDH:
            ID_list += [ID for ID,obj in self.load_dict.items() if not obj.is_DH]
        if flex:
            ID_list += [ID for ID,obj in self.load_dict.items() if obj.is_flex]
        if residential:
            ID_list += [ID for ID,obj in self.load_dict.items()]
        if office:
            ID_list += [ID for ID in self.office_dict.keys()]
            
        return ID_list

            

    ### ----- EFFICIENCY RELATED ---------------------------------------

    def introduce_efficiency(self, num = None, percent = 0.3):
        '''
        Simulating the efficiency trend. percent is the percentage
        of comsumption to reduce. And num is the number of loads to reduce.
        Percent indicates the reduction of consumption, 0.3 by default.
        '''

        self.is_efficient = True
        ID_list = self.get_loadID(residential = True,
                                  office = True)
        if num:
            ID_list = self.remove_list_element(ID_list, num = num)

        self.dataframe.loc[:,ID_list] = self.dataframe.loc[:, ID_list].apply(lambda x: x *(1-percent), axis = 0)
        





    
