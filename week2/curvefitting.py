# Imports at the top
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from typing import Iterable, List, Tuple


# Two new lines before functions
def read(path: str) -> pd.DataFrame:
    """Read in the NYTimes covid data
    Args:
        path (str): location of NYTimes  covid data
    Returns:
        pd.DataFrame: raw NYTimes covid data
    
    """
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df['county'] = df['county'].str.lower()
    df['state'] = df['state'].str.lower()
    df['cases'] = df['cases'].diff()
    df['cases'] = df['cases'].fillna(0)
    return df


def fit_daily_cases(df: pd.DataFrame, date: str):
    date_data = df.loc[df['date'] == pd.to_datetime(date), :]
    date_data = date_data['cases'].values
    date_data = date_data[date_data >= 0]
    date_data = date_data[date_data < 1000000]
    hist, bins = np.histogram(date_data, bins=20)
    hist = hist/np.sum(hist)  # Also can do hist/hist.sum()
    
    # Function we need is scipy.stats.poisson.pmf()
    pars, _ = scipy.optimize.curve_fit(scipy.stats.poisson.pmf, bins[:-1], hist, p0=[1])
    print(pars)

if __name__ == '__main__':
    df = read('data\\us_counties.csv')
    fit_daily_cases(df, '2022-01-05')