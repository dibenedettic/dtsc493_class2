#csv: comma separated variables
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
from typing import Iterable

def read(path:str) -> pd.DataFrame:
    """Read in csv file NYT data

    Args:
        path (str): Location of the file

    Returns:
        pd.DataFrame: Data frame for processing
    """
    df = pd.read_csv(path)
    df = df.loc[df['county'] == 'Allegheny']
    df = df.loc[df['state'] == 'Pennsylvania']
    df['cases'] = df['cases'].diff()
    df['cases'] = df['cases'].fillna(0)
    return df

def calc_recovered(df, pop_county = 1_250_000):
    """Calculate the recovered fraction

    Args:
        df (pd.DataFrame): NYTimes databased for a single county
        pop_county (int, optional): Population size for that county. Defaults to 1_250_000.

    Returns:
        float: fraction of recovered cases
    """
    return df['cases'].sum()/pop_county


def sir(s, i, r, r_naught, d, n_days):
    N = s + i + r
    gamma = 1/d
    beta = r_naught * gamma
    output = []
    #ds/dt, di/dt, dr/dt
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


def fit_sir(cases: pd.Series):
    cases = cases.values
    n = 1_250_578
    i = cases[0]
    s = n - cases[0]
    r = 0
    n_days = len(cases) - 1

    def fit_function(days, r0, d):
        return [v[1] for v in sir(s, i, r, r0, d, n_days)]

    best, _ = scipy.optimize.curve_fit(fit_function, range(n_days), cases, p0=[2, 10], bounds=[(0.5, 4), (5, 10)])
    return best

    
if __name__ == '__main__':

    #Anything after this will only be run if this file is the one that is run
    #Backslash is used for control characters
    #\n is newLine
    #\t is tab
    #Because \ is a separator and because it's a control character, we need to use \\ for every \
    #Libraries are collections of functions
    #Script is specific code to be executed
    df = read('data\\us_counties.csv')
    #print(df)
    df['cases'].plot()
    plt.show()
    fit_sir(df['cases'].iloc[200:270])


    