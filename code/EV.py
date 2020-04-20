import math
import random as rnd
rnd.seed(10) # for reproducibility
from collections import OrderedDict

import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
np.random.seed(1) # for reproducibility
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from spatialModelPkg.ev import EV
from spatialModelPkg.markov import Markov
from spatialModelPkg.simulation import Simulation
from spatialModelPkg.parkinglot import ParkingLot
from spatialModelPkg.extractDistances import extractDistances
from spatialModelPkg.extractFiles import readMatrixfiles

from spatialModelPkg.auxiliary import collect_stations_results

#-----------------------------

class EVStations:

    def __init__(self, ID, numberOfEVs, numberOfparkingloc, start, end, region):
        self.ID = ID
        self.region = region
        self.numberOfEVs = numberOfEVs
        self.numberOfparkingloc = numberOfparkingloc
        self.start = start
        self.end = end
        self.dataframe = self.create_EV_data()

    def create_EV_data(self):
        numberOfEVs = self.numberOfEVs
        numberOfparkingloc = self.numberOfparkingloc
        start = self.start
        end = self.end
        
        stationTypes = rnd.choices(range(3), k = numberOfparkingloc)
        stationsList = [(i, ParkingLot(ID = i,
                               state = stationTypes[i],
                               chargingPower = 3.7,
                               maximumOccupancy = numberOfEVs,
                               currentOccupancy = 0))
                    for i in range(numberOfparkingloc)]

        # add stations with no charging
        stationsList += [(str(i*1000), ParkingLot(ID = str(i*1000),
                               state = i,
                               chargingPower = 0.0,
                               maximumOccupancy = numberOfEVs,
                               currentOccupancy = 0,
                               chargingStatus = False)) for i in range(3)]
        stations = OrderedDict(stationsList)


        # create cars
        EVs = [EV(currentLocation = None,
                  currentState = None,
                  mpg = 0.2)
               for i in range (numberOfEVs)]

        # distribute the cars on the stations with state zero
        [x.inital_conditions(stations,0) for x in EVs]

        # load the weekday distances filter >200km
        weekdayDistances = extractDistances('../data/'+self.region+'/EV/distanceData/*day*.txt', 200)

        # load the weekend distances filter >200km
        weekendDistances = extractDistances('../data/'+self.region+'/EV/distanceData/*end*.txt', 200)

        # Create a distance Dictionary, if the key is true, use weekday distances,
        # else use weekend distances.
        dist = {True: weekdayDistances,
                False: weekendDistances}

        # WEEKDAY
        weekdayChain = readMatrixfiles('../data/'+self.region+'/EV/TransitionMatrix/*weekday*.txt')
        # define the transition Matrix
        weekday = Markov(weekdayChain)
        # WEEKEND
        weekdendChain = readMatrixfiles('../data/'+self.region+'/EV/TransitionMatrix/*weekend*.txt')
        # define the transition Matrix
        weekend = Markov(weekdendChain)

        # Create a Markov chain Dictionary, if the key is true, use weekday Markov chain,
        # else use weekend Markov chain.
        chain = {True: weekday,
                 False: weekend}

        # simulate one week
        minutes = pd.date_range(start, end, freq='min', tz='CET')[:-1]
        numberOfDays = (minutes[-1] - minutes[0]).days


        load = np.zeros(shape = (minutes.shape[0],
        len([v for (k,v) in stations.items() if v.chargingStatus == True])))

        # Setup the simulation Model
        simulationCase = Simulation(stations,
                                    EVs,
                                    chain,
                                    dist,
                                    minutes)

        # Estimate the electric load
        load = simulationCase.simulate_model()
        
        #ID_list = [str(i) for i in range(numberOfparkingloc)]
        #print(ID_list)
        #results = collect_stations_results(ID_list, load, stations)
        
        return pd.DataFrame(load, index=minutes).resample('H').sum()
