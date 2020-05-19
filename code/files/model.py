import pandas as pd
import numpy as np
import datetime

from substation import Substation
from net import Net
      
def GUI_substation(region,
                   resload_dict,
                   office_dict,
                   pv_dict,
                   num_EV,
                   num_parkingloc,
                   mpg_mu,
                   is_efficient,
                   efficiency_percent,
                   efficient_loads,
                   is_flex,
                   flex_days,
                   percent_flex_loads,
                   flex_reduction,
                   flex_only_noDH,
                   custom,
                   optimal_flex,
                   maxkW,
                   maxkWh):

    '''
    Function for creating a substation
    with attributes according to some
    scenario.

    Returns
    -------
    Substation object
    '''
    
    # Create substation
    substation = Substation(region)
    
    # Add residential loads to susbstation
    for load,num in resload_dict.items():
        if num >0:
            substation.add_residential_load(load,num)
        
    # Add offices
    for size,num in office_dict.items():
        if size > 0 and num > 0:
            substation.add_office(size= size,num = num)

    if not substation.start:
        substation.update_dates(start = datetime.datetime(2019, 1, 1),
                                end = datetime.datetime(2019, 12, 31))
    
    # Add PVs
    for size,num in pv_dict.items():
        if size > 0 and num > 0:
            substation.add_PV(size = size,
                              num = num)

    # Add custom loads from csv
    if not custom.empty:
        substation.add_custom(custom = custom)


    if not substation.dataframe.empty:
        substation.filter_whole_years(jan_start = True, num=1)

    # Add EVs after limiting to 1 year for computational reasons
    if num_EV > 0:
        substation.add_EV(num_EV, num_parkingloc, mpg_mu)

    # Introduce EE
    if is_efficient:
        substation.introduce_efficiency(percent = efficiency_percent,
                                        num = efficient_loads)
    # Introduce Optimal Flex
    if optimal_flex:
        substation.introduce_optimal_flex(maxkW, maxkWh)

    if not substation.dataframe.empty:
        substation.create_date_cols()
        substation.calculate_norm()
    
    # Introduce Flex
    if is_flex:
        substation.introduce_flexibility(days = flex_days,
                                         percent_loads = percent_flex_loads,
                                         reduction = flex_reduction,
                                         only_noDH = flex_only_noDH)
        
        
    return substation
