import pandas as pd
import numpy as np

class Office:

    weekdaylist = [25,25,25,25,25,25,
                   40,50,60,70,75,80,
                   80,80,75,70,60,40,
                   30,25,25,25,25,25]
    weekendlist = [25,25,25,25,25,25,
                   25,25,25,25,25,25,
                   25,25,25,25,25,25,
                   25,25,25,25,25,25]
        

    def __init__(self, ID, start, end, randomize = True):
        self.ID = ID
        self.start = start
        self.end = end
        self.mu = None
        self.sigma = None
        self.dataframe = pd.DataFrame(index = pd.date_range(start=start,
                                                            freq='H',
                                                            end=end))
        self.populate_dataframe()


    def populate_dataframe(self):
        self.dataframe['Weekday'] = self.dataframe.index.day_name()
        
        for index, row in self.dataframe.iterrows():
            loc = self.dataframe.index.get_loc(index)
            if self.dataframe.shape[0] - loc >= 24: 
                if row['Weekday'] in ['Monday', 'Tuesday','Wednesday','Thursday','Friday'] and index.hour in [0]:
                    self.dataframe.loc[self.dataframe.iloc[loc:loc+24].index,self.ID] = self.weekdaylist
                elif index.hour in [0]:
                    self.dataframe.loc[self.dataframe.iloc[loc:loc+24].index,self.ID] = self.weekendlist

        self.dataframe.drop('Weekday',axis = 1, inplace = True)
        self.dataframe.fillna(25, inplace = True)



    def description(self):
        self.calculate_norm()
        return ('An office load with '\
                + ' average consumption {} (-/+ {}) kWh per hour.'.format(self.mu,self.sigma))



    def be_random(self, sigma = 0.1, inplace = True):
        '''
        Introducing randomness to the office consumption
        through a stochastic deviation from the original.
        The values varies according to a gaussian
        distribution, with default and mu = 0, sigma = 0.1. 
        '''
        min_prob, max_prob = -sigma, sigma
        prob_array = (max_prob - min_prob) * np.random.random_sample(size=self.dataframe.shape[0]) + min_prob
        
        if inplace:
            self.dataframe[self.ID] += self.dataframe[self.ID].mul(prob_array)
        else:
            new_col_name = str(self.ID) + '_stoch_copy'
            self.dataframe[new_col_name] = self.dataframe[self.ID] + self.dataframe[self.ID].mul(prob_array)
        

    def calculate_norm(self):
        mu, sigma = scipy.stats.norm.fit(self.dataframe[self.ID].tolist())
        self.mu, self.sigma = round(mu,3), round(sigma,3)


        
