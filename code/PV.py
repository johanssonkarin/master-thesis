from matplotlib import pyplot as plt
import pandas as pd

class PV:
    '''
    Class representing PVs.
    Parameters: 
        size: 
            an int: 69/400/868 and represents m2.
        start:
            
    '''
    # Nameing here? e.g. 400 vs medium etc.
    path_dict = {69 : '../data/PV/pv69prod.csv' ,
                 400 : '../data/PV/pv400prod.csv',
                 868 : '../data/PV/pv868prod.csv'}

    def __init__(self, size, start, end, ID):
        self.size = size
        self.ID = ID
        self.start = start
        self.end = end
        self.mu = None
        self.sigma = None
        self.dataframe = pd.DataFrame(index = pd.date_range(start = start,
                                                            end = end, 
                                                            freq = 'H'),
                                      columns = ['Production'])
        self.populate_dataframe()

    def populate_dataframe(self):
        # Load the PV data as is
        csv_path = self.path_dict[self.size]
        pv_data = pd.read_csv(csv_path)
        ind = pd.date_range(start = self.start, periods = pv_data.shape[0], freq = 'H')
        pv_data.index = ind
        
        # Number of years of interest
        num_years = self.dataframe.index.year.nunique()
        first_year = self.dataframe.index[0].year

        # Generalize the data to all years of interest
        for i in range(0,num_years):
            # Below is a costum merge since the built-in is stupid hehe
            self.dataframe.loc[self.dataframe.index.isin(pv_data.index), pv_data.columns] = pv_data.loc[pv_data.index.isin(self.dataframe.index),pv_data.columns].values
            pv_data.index = pv_data.index.map(lambda x: x + pd.DateOffset(years=1))
        
        # Set missing values to 0 and convert to kWh
        self.dataframe.fillna(0, inplace = True)
        self.dataframe['Production'] /= 1000
        
    
    def calculate_norm(self):
        mu, sigma = scipy.stats.norm.fit(self.dataframe['Production'].tolist())
        self.mu, self.sigma = round(mu,3), round(sigma,3)
        
    def plot(self):
        #%matplotlib notebook
        plt.plot(self.dataframe)
        plt.title('Production of a {} m2 PV plant.'.format(self.size))
        plt.ylabel('Production [kWh]')
        
    def description(self):
        self.calculate_norm()
        return ('A {} m2 PV plant.'.format(self.size)\
                + ' The average production is of the plant is {} (-/+ {}) kWh per hour.'.format(self.mu,self.sigma))
        
        
    def be_random(self, sigma = 0.1, inplace = True):
        '''
        Introducing randomness to the PV production data through a 
        stochastic deviation from the original. The values varies 
        according to a gaussian distribution, 
        with default and mu = 0, sigma = 0.1. 
        '''
        min_prob, max_prob = -sigma, sigma
        prob_array = (max_prob - min_prob) * np.random.random_sample(size=self.dataframe.shape[0]) + min_prob
        new_col_name = 'Production' + '_stoch_copy'
        if inplace:
            self.dataframe['Production'] += self.dataframe['Production'].mul(prob_array)
        else:
            self.dataframe[new_col_name] = self.dataframe['Production'] + self.dataframe['Production'].mul(prob_array)
