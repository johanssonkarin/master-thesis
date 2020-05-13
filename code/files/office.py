import pandas as pd
import numpy as np
from scipy import stats

class Office:
    '''
    Class representing Office and Service buildings.
    
    Attributes
    -----------
        ID: int
            should be unique
        start: 
            string of date e.g. '2016-01-01'
        end:
            string of date e.g. '2016-01-01'
        reigon: str
            string of region e.g. 'Stockholm'
        size: int, optional
            800/12402/12667/28246/66799/73620
            represents atemp, i.e. heated
            area (m2) of the building. 800 by default.
            if size doesn't match a data set. A load
            is approximated using the other datasets.
        kwhp: float
            peak consumption of the load.
        kwhref: float
            kWh reference for kwh/year/m2.
        mu: float
            mean hourly consumption
        sigma: float
            standard deviation of the
            hourly consumption. 
        dataframe: pandas DataFrame
            time series data over consumption.
        
    '''
        

    def __init__(self, ID, start, end, region, size = 800):
        self.ID = ID
        self.size = size 
        self.start = start
        self.end = end
        self.region = region
        self.mu = 0
        self.sigma = 0
        self.kwhp = 0
        self.kwhref = 0
        self.populate_dataframe()
        self.calculate_stats()


    def populate_dataframe(self):
        '''
        Retrieving/constructing consumption data
        based on size.

        Returns
        -------
        None
        
        '''

        if self.size == 800:
            weekdaylist = [25,25,25,25,25,25,
                           40,50,60,70,75,80,
                           80,80,75,70,60,40,
                           30,25,25,25,25,25]
            weekendlist = [25,25,25,25,25,25,
                           25,25,25,25,25,25,
                           25,25,25,25,25,25,
                           25,25,25,25,25,25]

            self.dataframe = pd.DataFrame(index = pd.date_range(start=self.start,
                                                            freq='H',
                                                            end=self.end))
            self.dataframe['Weekday'] = self.dataframe.index.day_name()
            
            for index, row in self.dataframe.iterrows():
                loc = self.dataframe.index.get_loc(index)
                if self.dataframe.shape[0] - loc >= 24: 
                    if row['Weekday'] in ['Monday', 'Tuesday','Wednesday','Thursday','Friday'] and index.hour in [0]:
                        self.dataframe.loc[self.dataframe.iloc[loc:loc+24].index,self.ID] = weekdaylist
                    elif index.hour in [0]:
                        self.dataframe.loc[self.dataframe.iloc[loc:loc+24].index,self.ID] = weekendlist

            self.dataframe.drop('Weekday',axis = 1, inplace = True)
            self.dataframe.fillna(25, inplace = True)
        elif self.size == 12402:
            #separated since dataset is different and contains pv figures
            self.dataframe = pd.read_csv('../data/'+self.region+'/Office/office_12402_w_pv_wo.csv',
                                          index_col = 0,
                                          parse_dates = True)
            self.dataframe.drop(columns = ['PV','With'], inplace = True)
            self.dataframe.columns = [self.ID]
            
            #update start and end dates of the object according to data
            self.start = str(self.dataframe.index[0]).split(' ',1)[0]
            self.end = str(self.dataframe.index[0]).split(' ',1)[0]
        elif self.size in [12667,28246,66799,73620]: #sizes in dataset
            self.dataframe = pd.read_csv('../data/'+self.region+'/Office/office_'+str(self.size)+'.csv',
                                          index_col = 0,
                                          parse_dates = True)
            self.dataframe.columns = [self.ID]

            #update start and end dates of the object according to data
            self.start = str(self.dataframe.index[0]).split(' ',1)[0]
            self.end = str(self.dataframe.index[0]).split(' ',1)[0]
        else:
            #73620 most suitable for generalization
            self.dataframe = pd.read_csv('../data/'+self.region+'/Office/office_73620.csv',
                                          index_col = 0,
                                          parse_dates = True)
            self.dataframe.columns = [self.ID]
            self.dataframe[self.ID] = self.dataframe[self.ID].apply(lambda x: x/73260 * self.size)
            
            #update start and end dates of the object according to data
            self.start = str(self.dataframe.index[0]).split(' ',1)[0]
            self.end = str(self.dataframe.index[0]).split(' ',1)[0]


    def description(self):
        '''
        Short description of the office.

        Returns
        -------
        String
        
        '''
        self.calculate_stats()
        return ('An office load with '\
                + ' average consumption {} (-/+ {}) kWh per hour.'.format(self.mu,self.sigma))



    def be_random(self, sigma = 0.1, inplace = True):
        '''
        Introducing randomness to the office consumption
        through a stochastic deviation from the original.

        Parameters
        -----------
        sigma: float, optional
            The values varies according to a gaussian
            distribution, with mu = 0 and default sigma = 0.1.
        inplace: bool, optional
            performed inplace or if False, create
            a new column named 'ID_stoch_copy'
        
        Returns
        -------
        None
        
        '''
        min_prob, max_prob = -sigma, sigma
        prob_array = (max_prob - min_prob) * np.random.random_sample(size=self.dataframe.shape[0]) + min_prob
        
        if inplace:
            self.dataframe[self.ID] += self.dataframe[self.ID].mul(prob_array)
        else:
            new_col_name = str(self.ID) + '_stoch_copy'
            self.dataframe[new_col_name] = self.dataframe[self.ID] + self.dataframe[self.ID].mul(prob_array)
        self.calculate_stats()
        

    def calculate_stats(self):
        mu, sigma = stats.norm.fit(self.dataframe[self.ID].tolist())
        self.mu, self.sigma = round(mu,3), round(sigma,3)
        self.kwhp = self.dataframe[self.ID].max()
        self.kwhref = self.dataframe[self.ID].sum()/self.size #give correct numbers for full year atm



    def find_max(self):
        '''
        Returns date and time of largest
        hourly consumption of the office.

        Returns
        -------
        Timestamp. 
        '''
        return self.dataframe[self.ID].idxmax(axis = 0)
    



        



