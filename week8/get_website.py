from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
from typing import List
from googlesearch import search
import requests
import re
import numpy as np

from tenacity import retry_unless_exception_type


def scrape_website(path: str):
    """Return the string of the data from a website given a path

    Args:
        path (str): 
    
    Return:

    """
    #website = urllib.request.urlopen(path)
    website = requests.get(path)
    contents = website.text
    website.close()

    soup = BeautifulSoup(contents)
    [s.extract() for s in soup(['[document]', 'title'])]
    visible_text = soup.getText().replace('\r', '\n').replace('\t', '').lower().split('\n')

    visible_text = [t.strip() for t in visible_text if len(t.strip())]

    return visible_text

#evaluation features: url length(number of slashes), "asian" count, "southeast asian" count, "pacific islander" count, counts for every single country
# zipcode (or fips code), "aapi" count, 
# have to estimate recruiting area, then estimate demographics in recruiting area
# "organization"
# run evaluation from first five results of Google (diversity site:duq.edu)

def feature_url_length(url: str):
    """return number of slashes in url

    Args:
        url (str): _description_
    """
    counter = 0
    url = url.replace('https://', '').replace('http://', '')
    for letter in url:
        if letter == '/':
            counter += 1
    return counter

def generic_phrase_counter(txt: List[str], word: str):
    """Count the phrases in a list of strings

    Args:
        txt (list[str]): a list of strings as output from scraping
        word (str): a search word or phrase

    Return: a count of a search word or phrase
    """
    return sum([len(re.findall(r'\b%s\b' % word, phrase)) for phrase in txt])

def extract_all_features(url: str):
    """extract all features from a website regarding treatment of AAPI subgroups in diversity statements

    Args:
        url (str): url to query

    Return: dict(str,float):features as strings with counts as ints
    """
    full_scraped = scrape_website(url)
    scraped = [txt for txt in full_scraped if len(txt) > 40]
    out = {
        'url_length': feature_url_length(url),
        'count_asian': generic_phrase_counter(scraped, "asian"),
        'count_southeast_asian': generic_phrase_counter(scraped, "southeast asian"),
        'count_pacific_islander': generic_phrase_counter(scraped, "pacific islander"),
        'count_east_asian': generic_phrase_counter(scraped, "east asian"),
        'count_south_asian': generic_phrase_counter(scraped, "south asian"),
        'count_china': generic_phrase_counter(scraped, "china"),
        'count_india': generic_phrase_counter(scraped, "india"),
        'count_indonesia': generic_phrase_counter(scraped, "indonesia"),
        'count_pakistan': generic_phrase_counter(scraped, "pakistan"),
        'count_bangladesh': generic_phrase_counter(scraped, "bangladesh"),
        'count_japan': generic_phrase_counter(scraped, "japan"),
        'count_philippines': generic_phrase_counter(scraped, "philippines"),
        'count_vietnam': generic_phrase_counter(scraped, "vietnam"),
        'count_turkey': generic_phrase_counter(scraped, "turkey"),
        'count_iran': generic_phrase_counter(scraped, "iran"),
        'count_thailand': generic_phrase_counter(scraped, "thailand"),
        'count_myanmar': generic_phrase_counter(scraped, "myanmar"),
        'count_south_korea': generic_phrase_counter(scraped, "south korea"),
        'count_iraq': generic_phrase_counter(scraped, "iraq"),
        'count_afghanistan': generic_phrase_counter(scraped, "afghanistan"),
        'count_saudi_arabia': generic_phrase_counter(scraped, "saudi arabia"),
        'count_uzbekistan': generic_phrase_counter(scraped, "uzbekistan"),
        'count_malaysia': generic_phrase_counter(scraped, "malaysia"),
        'count_yemen': generic_phrase_counter(scraped, "yemen"),
        'count_nepal': generic_phrase_counter(scraped, "nepal"),
        'count_north_korea': generic_phrase_counter(scraped, "north korea"),
        'count_sri_lanka': generic_phrase_counter(scraped, "sri lanka"),
        'count_kazakhstan': generic_phrase_counter(scraped, "kazakhstan"),
        'count_syria': generic_phrase_counter(scraped, "syria"),
        'count_cambodia': generic_phrase_counter(scraped, "cambodia"),
        'count_jordan': generic_phrase_counter(scraped, "jordan"),
        'count_azerbaijan': generic_phrase_counter(scraped, "azerbaijan"),
        'count_uae': generic_phrase_counter(scraped, "united arab emirates"),
        'count_tajikistan': generic_phrase_counter(scraped, "tajikistan"),
        'count_israel': generic_phrase_counter(scraped, "israel"),
        'count_laos': generic_phrase_counter(scraped, "laos"),
        'count_lebanon': generic_phrase_counter(scraped, "lebanon"),
        'count_kyrgyzstan': generic_phrase_counter(scraped, "kyrgyzstan"),
        'count_turkmenistan': generic_phrase_counter(scraped, "turkmenistan"),
        'count_singapore': generic_phrase_counter(scraped, "singapore"),
        'count_kyrgyzstan': generic_phrase_counter(scraped, "kyrgyzstan"),
        'count_oman': generic_phrase_counter(scraped, "oman"),
        'count_palestine': generic_phrase_counter(scraped, "palestine"), #not sure if state of palestine
        'count_kuwait': generic_phrase_counter(scraped, "kuwait"),
        'count_georgia': generic_phrase_counter(scraped, "georgia"),
        'count_mongolia': generic_phrase_counter(scraped, "mongolia"),
        'count_armenia': generic_phrase_counter(scraped, "armenia"),
        'count_qatar': generic_phrase_counter(scraped, "qatar"),
        'count_bahrain': generic_phrase_counter(scraped, "bahrain"),
        'count_timor_leste': generic_phrase_counter(scraped, "timor-leste"),
        'count_cyprus': generic_phrase_counter(scraped, "cyprus"),
        'count_bhutan': generic_phrase_counter(scraped, "bhutan"),
        'count_maldives': generic_phrase_counter(scraped, "maldives"),
        'count_brunei': generic_phrase_counter(scraped, "brunei"),
        'count_taiwan': generic_phrase_counter(scraped, "taiwan"),
        'count_hong_kong': generic_phrase_counter(scraped, "hong kong"),
        'count_macao': generic_phrase_counter(scraped, "macao"),
        'zip': get_zipcode(full_scraped),
        'url': url
    }
    return out

def get_zipcode(txt: List[str]):
    zip_codes = []
    search_string = r'\D\d{5}\D' #we want a list of 5 digits
    #want re.search (re.match will find if there's a match)
    for text_element in txt:
        search = re.search(search_string, ' ' + text_element + ' ')
        if search is not None:
            zip_codes.append(search.group(0))
    return zip_codes[0][1:-1] if len(zip_codes) > 0 else None

def loop_through_universities(path:str):
    """_summary_

    Args:
        path (str): _description_
    """
    default_query = ' diversity and inclusion'
    universities = pd.read_csv(path)

    feature_list = []

    for university in universities['University']:
        try:
            query = university.lower() + default_query
            results = search(query, num=5, stop=5, pause=2)
            for i, result in enumerate(results): #enumerate gives us additonal variable that is the index
                feautures = extract_all_features(result)
                feautures['search_index'] = i
                feautures['university'] = university
                feautures['query'] = query 
                feature_list.append(feautures)
            print(university)    
        except:
            pass
    return pd.DataFrame(feature_list)

#How buried are the diversity statements
#what is the distribution mentions of specific groups ('asian' and subgroups)
#merging in data: merge together counts that are scraped, zip codes and zctas, aggregate by metro area, merge with scraped data

demo_col_names = { #subsetting column names in demo_df
    'NAME': 'zcta',
    'DP05_0001E' : 'pop_total',
    'DP05_0044PE' : 'percent_asian',
    'DP05_0045PE': 'percent_indian',
    'DP05_0046PE': 'percent_chinese',
    'DP05_0047PE' : 'percent_filipino',
    'DP05_0048PE' : 'percent_japanese',
    'DP05_0049PE': 'percent_korean',
    'DP05_0050PE': 'percent_vietnamese',
    'DP05_0051PE' : 'percent_other_asian',
    'DP05_0052PE': 'percent_nhpi',
    'DP05_0053PE': 'percent_nh',
    'DP05_0067PE': 'percent_twoplus_asian'
}

#re run this to get new combined.csv


zip_col_names = {
    'ZIP_CODE': 'zip',
    'PO_NAME': 'municipality',
    'STATE': 'state',
    'ZCTA': 'zcta'
}

def demographic_by_zip(zip_path: str, demo_path: str):
    """Load in zip code and demographic area data for merging

    Args:
        zip_path (str): _description_
        demo_path (str): _description_
    """
    zip_df = pd.read_csv(zip_path)
    zip_df = zip_df[[column for column in zip_col_names]].rename(columns=zip_col_names)
    demo_df = pd.read_csv(demo_path, skiprows=range(1,2))
    demo_df = demo_df[[column for column in demo_col_names]].rename(columns=demo_col_names)
    demo_df['zcta'] = demo_df['zcta'].str[5:].astype(int) #keeps zcta code without 'zcta5'
    
    zip_df = zip_df.loc[zip_df['zcta'] !='No ZCTA', :] # ', :' explicit about the rows and columns we want
    zip_df['zcta'] = zip_df['zcta'].astype(int)
    
    demo_df = demo_df.loc[demo_df['pop_total']!=0, :].astype(float)
    demo_df['zcta'] = demo_df['zcta'].astype(int)

    comb = zip_df.merge(demo_df, on='zcta', how='left')


    #agg to municipality level

    working_cols = ['pop_total', 'municipality', 'state'] + [column for column in demo_col_names.values() if column[:3] == 'per']

    for column in demo_col_names.values():
        if column[:3] == 'per':
            comb[column] = comb[column]*comb['pop_total']/100
    
    aggregated = comb[working_cols].groupby(['municipality', 'state']).transform('sum')
    
    for column in aggregated:
        if column[:3] == 'per':
            aggregated[column] = aggregated[column]/aggregated['pop_total']*100 #I think this was the error?
    

    comb = comb[['zcta', 'zip']].join(aggregated) #double brackets will produce a data frame and not a series
    # join() vs. merge(): merging on index not on columns

    return comb


def merge_dfs(demo_cross_df: pd.DataFrame, uni_df: pd.DataFrame):
    """Merging university dataframe with demographic dataframe

     Args:
         demo_df (pd.DataFrame): demographic dataframe
         uni_df (pd.DataFrame): university dataframe
    """
    def fill_nan_zip(sr):
        if sr.notna().sum()>0:
            return sr.loc[sr.first_valid_index()]
        else:
            return np.nan
    #groupby(search query).transform
    #goes outside
    uni_df['zip'] = uni_df[['university', 'zip']].groupby('university').transform(fill_nan_zip)['zip'].dropna().astype(int)
    comb2 = uni_df.merge(demo_cross_df, on='zip', how='left')
    #comb2['zip'] = comb2['zip'].dropna(subset = ['zip'])
    #comb2['zip'] = comb2['zip'].replace('', np.nan)
    comb2 = comb2.dropna(subset=['zip'])
    return comb2


if __name__ == "__main__":
    #df = loop_through_universities('data//good_university_list.csv')
    #df.to_csv('data//web_scraped_uni.csv')
    demo_cross_df = demographic_by_zip('data//zipzcta_crosswalk.csv', 'data//demo.csv')
    uni_df = pd.read_csv('data//web_scraped_uni.csv')
    comb = merge_dfs(demo_cross_df, uni_df)
    comb.to_csv('data//combined.csv')
#run new university csv
