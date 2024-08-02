from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator = IAMAuthenticator('aYwDWbWfxdlt8KgpZRj53VKJsofafjh80lVH9kxclTtX')
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)

nlu.set_service_url('https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/7e1fa7ba-2be7-478c-8265-ca795824491d')

def extract_keywords_from_url(url):
    response = nlu.analyze(
        url=url,
        features=Features(keywords=KeywordsOptions(limit=15))
    ).get_result()
    keywords = [keyword['text'] for keyword in response['keywords']]
    return keywords

def watson (user_url):
    response = nlu.analyze(
        url=user_url,
        features=Features(keywords=KeywordsOptions(limit=15))
    ).get_result()
    keywords = []
    for keyword in response['keywords']:
        if keyword['relevance'] > 0.50 and len(keywords) < 8:
            keywords.append(keyword['text'].encode('utf-8'))
    return keywords

url = 'https://indianexpress.com/article/india/pm-modi-obc-odisha-rahul-gandhi-bharat-jodo-nyay-yatra-9150657/'
keywords = extract_keywords_from_url(url)
print(keywords)
keywords = watson (url)
print(keywords)



