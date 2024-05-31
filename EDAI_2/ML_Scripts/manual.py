import pandas as pd
from subprocess import call
import re
# our own scripts
import ourModel as ourModel
import mlToOut as mlToOut

# INIT ALL ML
print("loading tensorflow  model")
sess, keep_prob_pl, predict, features_pl, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = ourModel.loadML()


def is_url(text):
    # Regular expression pattern to match URLs
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(url_pattern, text) is not None


def process_input(userInput):
    if is_url(userInput):
        print("Processing URL:", userInput)
        # Call function to process URL
        isURL = 'url'
        exit_code = call(["python", r"ML_Scripts\watson_scraper.py", "start", isURL, userInput])

    else:
        print("Processing claim:", userInput)
        # Call function to process claim 
        isURL = 'claim'
        exit_code = call(["python", r"ML_Scripts\watson_scraper.py", "start", isURL, userInput])

    if exit_code == 0:
        print("Execution successful")
    else:
        print("Execution failed")

    newsData = pd.read_csv(r'CSVs\url.csv')
    URLs = newsData['url'].tolist()
    SourceName = newsData['source'].tolist()
    BodyID = newsData['id'].tolist()

    # stances is a <List> of 0-3 classifications
    Stances = ourModel.runModel(sess, keep_prob_pl, predict, features_pl, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    # results = pd.DataFrame(Stances)
    # results.to_csv(r'CSVs\results.csv')
    BodyID = range(len(Stances))
    ml_output = pd.DataFrame(
        {'BodyID': BodyID,
        'Stances': Stances,
        'SourceName': SourceName,
        'URL': URLs
    })

    # making list of articles used as well as sending for averaging of stances
    response = ml_output.reset_index(drop=True)
    response = response.to_dict(orient='records')
    final_score = mlToOut.returnOutput(ml_output)
    final_score = (final_score + 1)/2
    print(f"final score: {final_score:.16f}")
    
    confidence = final_score
    if confidence > 0.5 and confidence < 0.7:
        resolve = 'likely to be'
    else:
        resolve = 'most likely'
    if final_score < 0.5:
        print (f"{resolve} Not Fake News")
    else:
        print (f"{resolve} Fake News")
    print (response)
    return response

if __name__ == '__main__':
    userInput = input("Enter a URL or a claim: ")

    # Process the input
    process_input(userInput)