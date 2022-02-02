#Imports at the top
import numpy as np
import pandas as pd
import scipy.optimize
from typing import Iterable, List, Tuple


#Two new lines before functions
def read(path:str):
    """Read in the NYTimes covid data
    
        Args: 
        path(str): location of NYTimes data
        
        Returns: 
        pd.DataFrame: raw NYTimes covid data

     """

    df = pd.read_csv(path)
    df['county'] = df['county'].str.lower()
    df['state'] = df['state'].str.lower()
    #df = df.loc[df['state'] == 'Pennsylvania']
    df['cases'] = df['cases'].diff()
    df['cases'] = df['cases'].fillna(0)
    return df


#Two lines before new functions
#except for class methods, where it's one line
def subset_county(df:pd.DataFrame, county: str, state: str, from_date: str, fips: int = None):
    """Subset the data to a specific county and time range

    Args:
        df (pd.DataFrame): raw NYTimes data
        county (str): county for subset, ignored if fips it set
        state (str): state for subset, ignored if fips is set
        from_date (str): first date to include
        fips (int, optional): county code, overrides county, state. Defaults to none. 

    Returns:
        pd.DataFrame: a subset of the original dataframe specific to a county and time range
    """
    if fips is None:
        df = df.loc[df['county'] == county.lower(), :]
        df = df.loc[df['state'] == state.lower(), :]
    else:
        df = df.loc[df['fips'] == fips, :]

    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[df['date'] >= pd.to_datetime(from_date), :]

    return df.reset_index(drop = True)


def sir(s, i, r, r_naught, d, n_days):
    """SIR Model that integrates over time

    Args:
        s (int): initial susceptible population
        i (int): initial infected population
        r (int): initial recover population
        r_naught (float): the r naught to use to propegate 
        d (float): the d to use to propegate (days of infection)
        n_days (int): number of days to propegate

    Returns:
        List[Tuple[float, float, float]]: A list of tuples of S,I,R for each day
    """
    N = s + i + r
    gamma = 1.0/d
    beta = r_naught * gamma
    output = []
    output.append([s,i,r])
    for day in range(n_days):
        dsdt = -(beta*i*s)/N
        didt = (beta*i*s)/N - gamma*i
        drdt = gamma*i
        s = dsdt + s
        i = didt + i
        r = drdt + r
        output.append([s,i,r])
    return output


def fit_rnaught(cases: pd.Series, population: int = 1250578):
    """Fit R0 and optionally d

    Args:
        cases (pd.Series): the column called cases from NYTimes df
        population (int): the population of the county in question

        Returns:
        float, float: the r_naught and d, respectively 

    """
    #do something similar with population?
    i = cases.to_list()[0]
    r = 0
    s = population - i - r
    def fit_function(x, r_naught):
        return np.diff([v[1] for v in sir(s,i,r,r_naught,10,len(x))])

    pars, _ = scipy.optimize.curve_fit(fit_function, np.arange(len(cases)-1), cases[1:].values, p0=[1.1], bounds=[[0.5],[15]])

    return pars[0]


#SCRIPT SECTION ==============================
if __name__ == '__main__':

    #Anything after this will only be run if this file is the one that is run
    #If imported, it will not be run.
    #Double-underscore/dunde methods are built in python

    #Backslash is used for control characters
    #\n is newLine
    #\t is tab
    #Because \ is a separator and because it's a control character, we need to use \\ for every \

    #Libraries are collections of functions
    #Script is specific code to be executed
    df = read('data\\us_counties.csv')
    allegheny = subset_county(df, 'allegheny', 'pennsylvania', '2021-12-26')
    print(allegheny)
    print(fit_rnaught(allegheny['cases']))
