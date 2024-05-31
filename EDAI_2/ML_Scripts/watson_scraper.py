from eventregistry import *
from threading import Thread, Lock
from py_ms_cognitive import PyMsCognitiveWebSearch
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import nltk
import pandas as pd
import sys

''' ibm_watson setup 
change Api key in authenticator and service URL '''

authenticator = IAMAuthenticator()
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url()

# event_registry setup
api_key = 
er = EventRegistry(apiKey = api_key)

global_df = pd.DataFrame()
mutex = Lock()
global_claims = []


# Given keywords, this funciton appends the article metadata to the global pandas dataframe
def get_articles(keyword):
    global global_df
    global global_claims
    local_claims = []

    q = QueryArticlesIter(keywords=QueryItems.AND(keyword))
    q.setRequestedResult(RequestArticlesInfo(count = 100, sortBy="sourceImportance"))

    x = 0

    local_df = pd.DataFrame()

    res = er.execQuery(q)
    for article in res['articles']['results']:
        if x == 0:
            news = article['title'].encode('utf-8')
        data = {
            'source': article['source']['title'].encode('utf-8'),
            'url' : article['url'].encode('utf-8'),
            'text' : article['body'].encode('utf-8')
        }
        local_df = pd.concat([local_df, pd.DataFrame(data,index=[x])])
        x += 1
    local_claims = [news] * (x)
    global_claims = global_claims + local_claims

    mutex.acquire()
    try:
        global_df = pd.concat([global_df,local_df])
    finally:
        mutex.release()

# Given a url, this function returns up to 15 keywords, keyword relevance might pose issue cause if
# relevance is set too high might get empty list lets set for 0.50 right now

def watson (url):
    response = nlu.analyze(
        url=url,
        features=Features(keywords=KeywordsOptions(limit=15))
    ).get_result()
    keywords = []
    for keyword in response['keywords']:
        if keyword['relevance'] > 0.50 and len(keywords) < 8:
            keywords.append(keyword['text'])
    return keywords

# Worker thread class override
class myThread(Thread):
    def __init__(self, query):
        Thread.__init__(self)
        self.query = query

    def run(self):
        get_articles(self.query)

# given claim, azure returns related urls using bing searches
def azure_search(claim):
    search_term = claim
    search_service = PyMsCognitiveWebSearch('75d1a40af4bf4ba4bdf561ae25b5db5c', claim)
    first_three_result = search_service.search(limit=3, format='json') #1-50

    urls = []
   # To get individual result json:
    for i in first_three_result:
        urls.append(i.url.encode('utf-8'))
    return urls

# given a list of urls, this function returns all related keywords for the urls
def azure_claim(urls):
    keywords = []
    for url in urls:
        keywords.append(watson(url))
    return keywords

# given keywords, query event registry and append to global dataframe
def watson_azure_scrape(keywords):
    global global_df

    index = 0
    threads = []

    for query in keywords:
        threads.append(myThread(query))
        threads[index].start()
        index += 1
    for thread in threads:
        thread.join()
    global_df = global_df.reset_index(drop=True)
    print (global_df.shape)

#     return global_df.to_dict(orient='records')

# Call this function with a claim to query event registry
def run_azure(claim):
    claim_tokens = nltk.word_tokenize(claim)
    if len(claim_tokens) == 3:
        # Go straight to event registry with claim
        watson_azure_scrape(claim)
    else:
        watson_azure_scrape(azure_claim(azure_search(claim)))

# Call this function with a url to query event registry
def watson_scrape(url):
    global global_df
    global global_claim
    print ("Getting keywords")
    keywords = watson(url)
    print (keywords)

    index = 0
    threads = []

    for query in keywords:
        threads.append(myThread(query))
        threads[index].start()
        index += 1
    for thread in threads:
        thread.join()
    global_df = global_df.reset_index(drop=True)
    global_df['id'] = range(len(global_df.index))
    bodies = global_df.loc[:,['id','text']]
    bodies.columns = ['BodyID','text']
    bodies.to_csv(r'CSVs\bodies.csv')
    headline = pd.DataFrame(global_claims)
    headline['BodyID'] = range(len(global_df.index))
    headline.columns = ['Headlines','BodyID']
    headline.to_csv(r'CSVs\claims.csv')
    urls = global_df.loc[:,['id','source','url']]
    urls.to_csv(r'CSVs\url.csv')
    return global_df.to_dict(orient='records')

def start(type_param, userInput):
    # Your logic to handle the parameters and perform the scraping
    if type_param == 'url':
        watson_scrape(userInput)
        print("got URL")
    else:
        run_azure(userInput)
        print("got Claim")

if __name__ == "__main__":
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 4:
        print("Usage: watson_scraper.py run <type_param> <input_param>")
        sys.exit(1)
    
    # Extract command-line arguments
    type_param = sys.argv[2]
    userInput = sys.argv[3]

    # Call the run function with the extracted parameters
    start(type_param, userInput)


