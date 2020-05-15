from residentialload import HouseNew, HouseOld, HouseDH, ApartmentNewDH
from office import Office
from EV import EVStations
from PV import PV

from matplotlib import pyplot as plt
from scipy import optimize as op
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
    This class represents an secondary substation.

    Parameters
    ----------
    region : str
        Location of the substation, e.g. 'Stockholm',
        needs to match data folder name.    
    
    
    Attributes
    ----------
    resload_dict: dict
        Contains all residential load objects.
        IDs are keys.
    PV_dict: dict
        Contains all PV objects.
        IDs are keys.
    office_dict: dict
        Contains all PV objects.
        IDs are keys.
    EV_list: list
        Contains ID of EV charging stations.
    custom_list: list
        Contains ID of Custom Loads
    dataframe: pandas DataFrame
        Every object is a column. DateTimeIndex.
    ID_count: int
        Keeps track of IDs.
    resload_count: int
        Number of residential loads.
    house_count: int
        Number of residential houses.
    apartment_count: int
        Number of appartments.
    office_count: int
        Number of offices.
    flex_count:
        Number of flexible agents.
    custom_count: int
        Number of custom loads.
    DH_count: int
        Number of loads with district heatning.
    PV_count: int
        Number of PV plants.
    EV_count: int
        Number of EVs in the Substation.
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
    percentile_dict: dict
        A dictionary of percentiles and associated
        values of the aggregated load.
    start: str
        Start date of the time series data.
    end: 
        End date of the time series data.
    
    '''
    
    
    # Initializer / Instance Attributes
    def __init__(self, region):
        self.resload_dict = dict()
        self.PV_dict = dict()
        self.office_dict = dict()
        self.EV_list = list()
        self.custom_list = list() 
        self.dataframe = pd.DataFrame()
        self.ID_count = 0
        self.resload_count = 0
        self.house_count = 0
        self.apartment_count = 0
        self.office_count = 0
        self.flex_count = 0
        self.DH_count = 0
        self.PV_count = 0
        self.EV_count = 0
        self.custom_count = 0
        self.region = region
        self.is_flex = False
        self.is_efficient = False
        self.mu = None
        self.sigma = None
        self.percentile_dict = None
        self.start = None
        self.end = None
        self.coldest_days = []
        self.region_path = '../data/'+region+'/temperature.csv'



    ###--------------- ADD LOADS ------------------------------
        
    def add_residential_load(self, load_type, num = 1):
        '''
        Method for adding residential builings.
        
        Parameters
        ----------
        load_type: str
            Which type to add. Representing a residential
            load class.
        num: int
            Number of loads.

        Returns
        -------
        None.
        
        '''
        if num > 0:
            if load_type == 'HouseNew':
                for i in range(0,num):
                    self.ID_count += 1
                    self.resload_count += 1
                    self.house_count += 1
                    load = HouseNew(region = self.region, ID = self.ID_count)
                    self.resload_dict[self.ID_count] = load 
                    
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
                    self.resload_count += 1
                    self.house_count += 1
                    load = HouseOld(region = self.region, ID = self.ID_count)
                    self.resload_dict[self.ID_count] = load 
                    
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
                    self.resload_count += 1
                    self.house_count += 1
                    self.DH_count += 1
                    load = HouseDH(region = self.region, ID = self.ID_count)
                    self.resload_dict[self.ID_count] = load 
                    
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
                    self.resload_count += 1
                    self.apartment_count += 1
                    self.DH_count += 1
                    load = ApartmentNewDH(region = self.region, ID = self.ID_count)
                    self.resload_dict[self.ID_count] = load 
                    
                    if self.dataframe.empty:
                        self.dataframe = load.dataframe
                    else:
                        self.dataframe = self.dataframe.merge(load.dataframe,
                                                              how = 'inner',
                                                              left_index=True,
                                                              right_index=True)
            self.update_dates(self.dataframe.index[0],self.dataframe.index[-1])

            
    def add_office(self, size, num = 1, randomize = True, is_DH = True):
        '''
        Adding office objects to the substation.

        Parameters
        ----------
        size: int
            800/12402/12667/28246/66799/73620
            represents atemp, i.e. heated
            area (m2) of the building. 800 by default.
            if size doesn't match a data set. A load
            is approximated using the other datasets.
        num: int
            number of offices of that size to add.
        randomize: bool, optional
            True by default. Offices are randomized
            by a percentage drawn from a gaussion distribution
            with mu = 0 and sigma = 0.1.
        is_DH: bool, optional
            If the office is district heated or not. Effects
            the DH_count of the substation.

        Returns
        -------
        None.
        
        '''
        if num>0:
            for i in range(0,num):
                self.ID_count += 1
                self.office_count +=1
                if is_DH:
                    self.DH_count +=1

                office = Office(size = size,
                                start = self.start,
                                end = self.end,
                                ID = self.ID_count,
                                region = self.region,
                                )

                # Save office object to dict
                self.office_dict[office.ID] = office
                
                if randomize:
                    office.be_random()

                # Add office to substation dataframe
                if self.dataframe.empty:
                            self.dataframe = office.dataframe
                else:
                    self.dataframe = self.dataframe.merge(office.dataframe, 
                                                          how = 'inner',
                                                          left_index=True,
                                                          right_index=True)
            self.update_dates(self.dataframe.index[0],self.dataframe.index[-1])
        

    def add_PV(self, size, num = 1, randomize = True):
        '''
        Method for adding PV plants to the substation.

        Parameters
        -----------
        size: int
            Refers to installed m2 of the PV.
        num: int
            Number of PV plants to add of the specified size.
        randomize: bool, optional
            If the production profile should be randomized by
            a percentage drawn from a gaussion distribution
            (mu = 0, sigma = 0.1).

        Returns
        -------
        None.
        '''
        if num > 0:
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
                self.PV_dict[pv.ID] = pv
                
                # PV production means neagtive consumption. Merge negative values
                pv_df_neg = pv.dataframe
                pv_df_neg[pv.ID] *= (-1)
                if self.dataframe.empty:
                    self.dataframe = pv_df_neg
                else:
                    self.dataframe = self.dataframe.merge(pv_df_neg,
                                                          how = 'inner',
                                                          left_index=True,
                                                          right_index=True)
            self.update_dates(self.dataframe.index[0],self.dataframe.index[-1])

    def add_EV(self, num_EV, num_parkingloc, mpg_mu = 0.2, mpg_sigma = 0.05):
        '''
        Method for adding EV charging stations.
        
        Parameters
        ----------
        num_EV: int
            Number of EVs to charge.
        num_parkingloc: int
            Number of locations to charge the EVs.

        Returns
        -------
        None.
        
        '''
        self.EV_count += num_EV
        
        ev = EVStations(numberOfEVs = num_EV,
                        numberOfparkingloc = num_parkingloc,
                        start = self.start,
                        end = self.end,
                        region = self.region,
                        mpgMu = mpg_mu,
                        mpgSigma = mpg_sigma
                        )

        #rename the charging stations according to
        # their IDs. for compatability :)
        ev.dataframe.columns = [i for i in range(self.ID_count+1,self.ID_count+num_parkingloc+1)]

        # Add IDs to list of EV IDs to keep track
        self.EV_list += [i for i in range(self.ID_count+1,self.ID_count+num_parkingloc+1)]
        self.ID_count += num_parkingloc
        if self.dataframe.empty:
            self.dataframe = ev.dataframe
        else:
            self.dataframe = self.dataframe.merge(ev.dataframe,
                                                  how = 'inner',
                                                  left_index=True,
                                                  right_index=True)
        self.update_dates(self.dataframe.index[0],self.dataframe.index[-1])


    def add_custom(self, csv_path = None, custom = None): 
        '''
        Method for adding loads which are not in
        any of the datasets of the model by passing a
        local csv path or a dataframe.

        Parameters
        ----------
        csv_path: str, optional
            Path to a local compatible csv-file.
        custom: pandas DataFrame, optional
            Should contain time series of consumption in
            kWh and have a DateTimeIndex.
            Could be one or more loads.

        Returns
        -------
        None.
        '''

        if csv_path:
            self.ID_count += 1
            # Create dataframe
            custom = pd.read_csv(csv_path, index_col = 0, parse_dates = True)
        if custom:
            # Rename columns (+1 or not?)
            custom.columns = [range(self.ID_count, self.ID_count+custom.shape[1])]
            # Add to substation dataframe
            if self.dataframe.empty:
                self.dataframe = custom
            else: 
                self.dataframe = self.dataframe.merge(custom, 
                                                  how = 'inner',
                                                  left_index=True,
                                                  right_index=True)

            # Remember IDs and number of loads
            self.custom_list += [range(self.ID_count, self.ID_count+custom.shape[1])]
            self.custom_count += custom.shape[1]           
        
        self.update_dates(self.dataframe.index[0],self.dataframe.index[-1])
        

    ### ----------- SUBSTATION RELATED -------------------------------
        
    def update_dates(self, start, end):
        self.start = start
        self.end = end
        
    def calculate_norm(self, percentiles = [1, 25, 50, 75, 90, 99] ):
        self.update_aggregated_col()
        
        values = np.percentile(self.dataframe['AggregatedLoad'], q=percentiles)
        self.percentile_dict = dict(zip(percentiles, values))
        
        mu, sigma = scipy.stats.norm.fit(self.dataframe['AggregatedLoad'].tolist())
        self.mu, self.sigma = round(mu,3), round(sigma,3)
        
        
    def create_date_cols(self):
        '''
        Add information about each index in
        columns, including year, month, weekday
        and hour.

        Returns
        -------
        None. 
        '''
        self.dataframe['Year'] = self.dataframe.index.year
        self.dataframe['Month'] = self.dataframe.index.month
        self.dataframe['Weekday'] = self.dataframe.index.day_name()
        self.dataframe['Hour'] = self.dataframe.index.hour


    def find_max(self):
        '''
        Returns date and time of largest
        hourly consumption of the substation.

        Returns
        -------
        Timestamp. 
        '''
        if self.dataframe.empty: # edge case
            return 'No max'
        
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
               + ' The substation contains {} loads with an '.format(self.resload_count)\
                + 'aggregated average comsumption of {} (-/+ {}) kWh per hour.'.format(self.mu,self.sigma))
    
    

    def filter_whole_years(self, jan_start = True, num = None):
        '''
        Method for limiting substation dataframe to whole years.
        By default jan-dec but can be changed to whole
        years from first date index. The 'num'
        parameter specifies number of years to keep
        and needs to minimum 1. If num is not specified,
        maximum number of years are kept. 
        '''
        #first_date, last_date = self.start, self.end
        first_date, last_date = self.dataframe.index[0], self.dataframe.index[-1]

        if num:
            if jan_start:
                if first_date.is_year_start:
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
                if first_date.is_year_start:
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



    def get_load_type(self,ID):
        '''
        Method for retriving class name
        of a load.

        Parameters
        ----------
            ID: int
                Corresponding to an ID of a
                load within the substation.
                
        Returns
        -------
        String.

        '''
        # Check if residential
        if ID in self.resload_dict.keys():
            return self.resload_dict[ID].__class__.__name__
        # Check if Office
        if ID in self.office_dict.keys():
            return self.office_dict[ID].__class__.__name__
        # Check if PV plant
        if ID in self.PV_dict.keys():
            return self.PV_dict[ID].__class__.__name__
        # This may not be correct and needs fixing
        # and EV charge should come first since
        # > ID_count doesn't always hold true. 
        if ID in self.EV_list:
            return 'EVChargingStation'
        if ID in self.custom_list:
            return 'Custom'
        else:
            return 'There is no load with {} as ID in the substation.'.format(ID)
        

    def get_loadID(self,
                   new = False,
                   old = False,
                   noDH = False, 
                   flex = False,
                   residential = False,
                   office = False):
        '''
        Method that returns ID of loads of the substation
        based on attribute.
        '''
        ID_list = []
        
        if noDH:
            ID_list += [ID for ID,obj in self.resload_dict.items() if not obj.is_DH]
        if new:
            ID_list += [ID for ID,obj in self.resload_dict.items() if obj.is_new]
        if old:
            ID_list += [ID for ID,obj in self.resload_dict.items() if not obj.is_new]
        if flex:
            ID_list += [ID for ID,obj in self.resload_dict.items() if obj.is_flex]
        if residential:
            ID_list += [ID for ID,obj in self.resload_dict.items()]
        if office:
            ID_list += [ID for ID in self.office_dict.keys()]
            
        return ID_list
    
    
    def plot_single_load(self, ID, start = None, end = None):
        '''
        Plots the load curve of a single load.

        Parameters
        ----------
            ID: int
                The ID of the load to plot.
            start: str, optional
                e.g. '2019-01-01'. if
                not provided the start date 
                of the dataframe is used
            end: str, optional
                e.g. '2019-01-02'. if
                not provided the end date
                of the dataframe is used

        Returns
        -------
        Matplotlib object
        
        '''
        # Add resample and freq??

        if not start: #use dataframe start index if no input
            start = self.start
        if not end: #use dataframe end index if no input
            end = self.end

        load_name = self.get_load_type(ID)
        #cut dataframe to time period of interest
        data = self.dataframe[start:end]
        load_plt = plt.plot(data[ID])
        plt.title('Load curve of type ´{}´ with ID {}'.format(load_name,ID))
        plt.ylabel('kWh')
        return load_plt



    def plot_load_duration_curve(self):
        '''
        Method that takes a sorted list of load demand 
        values and produces a plot of the load duration curve.
        '''
        sorted_demand_list = self.dataframe['AggregatedLoad'].sort_values(ascending=False).tolist()
        list_len = len(sorted_demand_list) # Number of datapoints
        x = np.linspace(1,list_len,list_len).tolist() # List of hours

        fig = plt.plot(x,sorted_demand_list)
        plt.title('Load duration curve')
        plt.xlabel('Hours')
        plt.ylabel('kWh')

        return fig


    def print_insights(self, 
                       duration_curve = True,
                       month_plot = True, 
                       weekday_plot = True, 
                       hour_plot = True):
        '''
        Method for generating and printing different kinds of
        information about the dataframe and load profiles. 
        '''
        
        self.update_aggregated_col()
        if 'Month' not in self.dataframe.columns: 
            self.create_date_cols()
            
        if duration_curve:
            return self.plot_load_duration_curve()
    
        if month_plot:
            ax = sns.boxplot(data=self.dataframe, x='Month', y='AggregatedLoad')
            ax.set_ylabel('kWh')
            ax.set_title('Hourly consumption of the substation')
        
        if weekday_plot:
            sns.set(style="whitegrid")
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            ax1 = sns.boxplot(data=self.dataframe, x='Weekday', y='AggregatedLoad', order=order)
            ax1.set_ylabel('kWh')
            ax1.set_xlabel('')
            ax1.set_title('Hourly consumption of the substation')
            
        if hour_plot:
            ax = sns.boxplot(data=self.dataframe, x='Hour', y='AggregatedLoad')
            ax.set_ylabel('kWh')
            ax.set_title('Hourly consumption of the substation')


            

    ### ----- 'STATIC' FLEX ---------------------------------------
    
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
        if 'AggregatedLoad' not in self.dataframe:
            self.update_aggregated_col()
        self.dataframe['NoFlex'] = self.dataframe['AggregatedLoad']
        if not self.coldest_days:
            self.find_coldest_days(days)
        
        if only_noDH:
            ID_list = self.get_loadID(noDH = True)
        else:
            ID_list = self.get_loadID(residential = True) #get all IDs

        ID_list = self.remove_list_element(ID_list, percentage = percent_loads)
        for ID in ID_list:
            self.flex_count += 1
            self.resload_dict[ID].be_flexible(self.coldest_days, reduction)
            self.dataframe.loc[:,ID] = self.resload_dict[ID].dataframe
            #self.dataframe.loc[self.dataframe.index.isin(self.load_dict[ID].dataframe.index), self.load_dict[ID].dataframe.columns] = self.load_dict[ID].dataframe.loc[self.load_dict[ID].dataframe.index.isin(self.dataframe.index), self.load_dict[ID].dataframe.columns].values
        self.update_aggregated_col()
        

    def remove_list_element(self, thelist, percentage = None, num = None):
        '''
        Removing a random number of elements or a percentage
        of element from a list.

        Parameters
        ----------
        thelist: list
            The list from which elements should be removed.
        percentage: float, optional
            [0,1] percentage to remove.
        num: int, optional
            Number of elements to remove.

        Returns
        -------
        List.
        
        '''
        random.shuffle(thelist)
        if percentage:
            count = int(len(thelist) * percentage)
        elif num:
            count = num
        if not count: return []  # edge case, no elements removed
        return thelist[-count:]        

        
    
    def find_coldest_days(self, num):
        '''
        Method for finding the coldest days per year in a region, which is then updated 
        in the substation attribute 'coldest_days'. The path to the 
        temperature data needs to be specified within the class
        attribute 'region_path_dict'. The start and end of the timeframe to
        check is determined by the start and end attributes of the object.

        Parameters
        ----------
        num: int
            The number of coldest days/year to find.

        Returns
        -------
        None.
        
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


            

    ### ----- EFFICIENCY  ----------------------------------------------

    def introduce_efficiency(self, num = None, percent = 0.3):
        '''
        Simulating the efficiency trend. percent is the percentage
        of comsumption to reduce. And num is the number of loads to reduce.
        Percent indicates the reduction of consumption, 0.3 by default.
        '''

        self.is_efficient = True
        ID_list = self.get_loadID(residential = True, #Which loads should be effected?
                                  office = True)        # Now all office + res. Use get_loadID 
        if num:
            ID_list = self.remove_list_element(ID_list, num = num)

        self.dataframe.loc[:,ID_list] = self.dataframe.loc[:, ID_list].apply(lambda x: x *(1-percent), axis = 0)
        


    ### ----- OPTIMAL FLEX --------------------------------------------------
    
    def introduce_optimal_flex(self, maxkW, maxkwWh, optimize_months = [10,11,12,1,2,3]):
        '''
        Method for including a additional column of optimized
        aggregated load, given some contraints. Currently
        addressing substation level and a time window of
        24 hours / a day.

        Parameters
        ----------
        maxkW: float
            How much of the load that could be reduced during
            one hour, i.e. reduction of one value. 
        maxkWh: float
            Total reduction of energy during the specified
            time frame.
        optimize_months: list of ints, optional
            Which months to optimize, summer/winter
            is the current optimization criteria. 
            
        Returns
        --------
        None.
        '''
        # Copying the consumption to later replace some 
        # chosen values with optimal values
        self.dataframe['OptimalLoad'] = self.dataframe['AggregatedLoad']

        # Optimizing per day
        index_days = pd.date_range(start = self.dataframe.index[0],
                                   end = self.dataframe.index[-1],
                                   freq = 'D')

        # Optimize all days during winter months
        for index in index_days:
            if index.month in optimize_months:
                consumption = self.dataframe[str(index).split(' ',1)[0]]['AggregatedLoad'].tolist()
                self.dataframe.loc[str(index).split(' ',1)[0],'OptimalLoad'] = self.optimize_consumption(consumption,
                                                                                                         maxkW,
                                                                                                         maxkwWh)

    def optimize_consumption(self, actual_consumption, maxkW, maxkWh):
        # inequalities in the form f(x) >= 0
        constraints = ({'type': 'ineq','fun': lambda x: maxkW - abs(max(x))}, # max recover / flex limited by maxMW
                        {'type': 'ineq','fun': lambda x: maxkWh - sum(abs(x[x<0]))}, #max flex energy limited by maxMWh
                        {'type': 'ineq','fun': lambda x: -sum(x)}, # total response cannot be positive, i.e. customer cannot consume more than baseline
                        #{'type': 'ineq','fun': lambda x: np.ones(self.slack)*flex_left_to_recover - np.dot(np.tril(np.ones((self.slack,self.slack))),x)},
                        #{'type':'ineq','fun':lambda x: 0 - abs(sum(x)))}
                        )
        # daily time window --> 24 values    
        x0 = np.zeros(24) 
        # minimize max given the above contraints
        results = op.minimize(lambda x: self.flex_function(actual_consumption,x), x0, constraints = constraints)
        #return the optimized results
        return [sum(x) for x in zip(actual_consumption, results.x)] 
        
        # subfunction to optimize
    def flex_function(self, actual_consumption, x):
        return max(actual_consumption + x)
        
