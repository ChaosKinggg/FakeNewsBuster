from eventregistry import *
import pandas as pd

api_key = '492ae2f5-4d0a-445c-8063-a96c5876c1e2'
er = EventRegistry(apiKey = api_key)

global_claim = ''

def get_articles(keyword):
    global global_claim
    q = QueryArticlesIter(keywords=QueryItems.AND(keyword))
    print ('made query')
    q.setRequestedResult(RequestArticlesInfo(count= 100, sortBy="sourceImportance"))
    # throws stupid errors if tried to fetch beyound 100 articles for a single querry
    print ('setted requested results')
    x = 0

    local_df = pd.DataFrame()

    res = er.execQuery(q)
    print ('got results')
    for article in res['articles']['results']:
        if x == 0:
            global_claim = article['title'].encode('utf-8')
        data = {
            'source': article['source']['title'].encode('utf-8'),
            'url' : article['url'].encode('utf-8'),
            'text' : article['body'].encode('utf-8')
        }
        local_df = pd.concat([local_df, pd.DataFrame(data,index=[x])])
        x += 1
    return local_df

print ('start')
print (get_articles(keyword='Prime Minister Narendra Modi'))

'''

['Congress leader Rahul Gandhi', 'Prime Minister Narendra Modi', 'caste census', 'Congress leader',
 'per cent tribals', 'last year\xe2\x80\x99s Bharat Jodo Yatra',
   'Union minister Pralhad Joshi', 'Rahul Gandhi']

'''
