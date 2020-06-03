from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats

class PV:
    '''
    Class representing PVs.

    Attributes
    -----------
    size: 
        an int: 69/400/868 or custom. Represents m2
        of the panels.
    reigon:
        string of region e.g. 'Stockholm'
    start:
        string of date e.g. '2016-01-01'
    end:
        string of date e.g. '2016-01-01'
    ID:
        int which should be unique
            
    '''

    def __init__(self, region, size, start, end, ID):
        self.size = size
        self.ID = ID
        self.customsize = False
        self.start = start
        self.end = end
        self.mu = None
        self.sigma = None
        self.dataframe = pd.DataFrame(index = pd.date_range(start = start,
                                                            end = end, 
                                                            freq = 'H'),
                                      columns = [self.ID])
        self.path_dict = {69 : '../data/' + region + '/PV/pv69prod.csv' ,
                          400 : '../data/' + region + '/PV/pv400prod.csv',
                          868 : '../data/' + region + '/PV/pv868prod.csv'}
        self.populate_dataframe()

    def populate_dataframe(self):
        '''
        Auxillary method for generating the PV object,
        which populates the dataframe attribute.

        Returns
        -------
        None.
        
        '''
        
        # Load the PV data as is
        if self.size in self.path_dict.keys():
            csv_path = self.path_dict[self.size]
        else:
            self.customsize = True
            csv_path = self.path_dict[69]
        pv_data = pd.read_csv(csv_path)
        pv_data.columns = [self.ID]
        ind = pd.date_range(start = self.start, periods = pv_data.shape[0], freq = 'H')
        pv_data.index = ind
        
        # Number of years of interest
        num_years = self.dataframe.index.year.nunique()

        # Generalize the data to all years of interest
        for i in range(0,num_years):
            self.dataframe = self.dataframe.combine_first(pv_data) #fills NaN values using pv-dataframe
            pv_data.index = pv_data.index.map(lambda x: x + pd.DateOffset(years = 1))
        
        # Set missing values to 0 and convert to kWh
        self.dataframe.fillna(0, inplace = True)
        self.dataframe[self.ID] /= 1000

        if self.customsize:
            self.dataframe[self.ID] = self.dataframe[self.ID].apply(lambda x: (x/10.5) * self.size)
        
    
    def calculate_norm(self):
        '''
        Method which calculates mean and standard
        deviation of the dataframe attribute.

        Returns
        -------
        None.

        '''
        
        mu, sigma = scipy.stats.norm.fit(self.dataframe[self.ID].tolist())
        self.mu, self.sigma = round(mu,3), round(sigma,3)
        
    def plot(self):
        '''
        Method generating a plot over
        production curve of the pv object.

        Returns
        -------
        None.

        '''
        #%matplotlib notebook
        plt.plot(self.dataframe)
        plt.title('Production of a {} m2 PV plant.'.format(self.size))
        plt.ylabel('Production [kWh]')


    def plot_month(self):
        '''
        Method generating a boxplot over
        production of the pv object, categorized by month.

        Returns
        -------
        None.

        '''
        self.dataframe['Month'] = self.dataframe.index.month
        sns.set(style="whitegrid")
        ax = sns.boxplot(data=self.dataframe, x='Month', y=self.ID)
        ax.set_ylabel('kWh')
        ax.set_title('Hourly production of {} m2 PV plant'.format(self.size))
        
    def description(self):
        '''
        Method generating a minimal description of
        the pv objects.

        Returns
        -------
        String.

        '''
        self.calculate_norm()
        return ('A {} m2 PV plant.'.format(self.size)\
                + ' The average production is of the plant is {} (-/+ {}) kWh per hour.'.format(self.mu,self.sigma))
        
        
    def be_random(self, sigma = 0.1, inplace = True):
        '''
        Method which randomness to the PV production data through a 
        stochastic deviation from the original.

        Parameters
        ----------
        sigma: float, optional
            Indicates the varitation of the deviation
            according to a gaussian distribution, by default 0.1.
        inplace: bool, optional
            If the new random values should replace the old
            or if a new column should be created within the dataframe.

        Returns
        -------
        None.
        '''
        min_prob, max_prob = -sigma, sigma
        prob_array = (max_prob - min_prob) * np.random.random_sample(size=self.dataframe.shape[0]) + min_prob
        if inplace:
            self.dataframe[self.ID] += self.dataframe[self.ID].mul(prob_array)
        else:
            new_col_name = str(self.ID) + '_stoch_copy'
            self.dataframe[new_col_name] = self.dataframe[self.ID] + self.dataframe[self.ID].mul(prob_array)
