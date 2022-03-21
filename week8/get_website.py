from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
from typing import List
from googlesearch import search
import requests
import re


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
    return sum([phrase.count(word) for phrase in txt])

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
        'zip': get_zipcode(full_scraped)
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
    return zip_codes[0] if len(zip_codes) > 0 else None

def loop_through_universities(path:str):
    """_summary_

    Args:
        path (str): _description_
    """
    default_query = ' diversity and inclusion'
    universities = pd.read_csv(path)

    feature_list = []

    for university in universities['university']:
        try:
            query = university.lower() + default_query
            results = search(query, num=5, stop=5, pause=2)
            for i, result in enumerate(results): #enumerate gives us additonal variable that is the index
                feautures = extract_all_features(result)
                feautures['search_index'] = i
                feautures['university'] = university
                feautures['url'] = query
                feature_list.append(feautures)
            print(university)    
        except:
            pass
    return pd.DataFrame(feature_list)


if __name__ == "__main__":
    df = loop_through_universities('data//university_list.csv')
    df.to_csv('data//web_scraped_uni.csv')

    