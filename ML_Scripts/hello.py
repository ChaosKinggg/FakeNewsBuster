import pandas as pd
from subprocess import call
import re
import tkinter as tk
from tkinter import messagebox
# our own scripts
import ourModel as ourModel
import mlToOut as mlToOut

# Initialize ML model
print("loading tensorflow model")
sess, keep_prob_pl, predict, features_pl, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = ourModel.loadML()

def is_url(text):
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(url_pattern, text) is not None

def process_input(userInput):
    if is_url(userInput):
        print("Processing URL:", userInput)
        isURL = 'url'
        exit_code = call(["python", r"ML_Scripts/watson_scraper.py", "start", isURL, userInput])
    else:
        print("Processing claim:", userInput)
        isURL = 'claim'
        exit_code = call(["python", r"ML_Scripts/watson_scraper.py", "start", isURL, userInput])

    if exit_code == 0:
        print("Execution successful")
    else:
        print("Execution failed")

    newsData = pd.read_csv(r'CSVs/url.csv')
    URLs = newsData['url'].tolist()
    SourceName = newsData['source'].tolist()
    BodyID = newsData['id'].tolist()

    Stances = ourModel.runModel(sess, keep_prob_pl, predict, features_pl, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    BodyID = range(len(Stances))
    ml_output = pd.DataFrame({'BodyID': BodyID, 'Stances': Stances, 'SourceName': SourceName, 'URL': URLs})

    response = ml_output.reset_index(drop=True)
    response = response.to_dict(orient='records')
    final_score = mlToOut.returnOutput(ml_output)
    final_score = (final_score + 1) / 2
    print(f"final score: {final_score:.16f}")

    confidence = final_score
    if 0.5 < confidence < 0.7:
        resolve = 'likely to be'
    else:
        resolve = 'most likely'
    if final_score < 0.5:
        result = f"{resolve} Not Fake News"
    else:
        result = f"{resolve} Fake News"
    print(response)
    return result, response

def on_submit():
    userInput = entry.get()
    result, response = process_input(userInput)
    messagebox.showinfo("Result", result)

# Initialize Tkinter
root = tk.Tk()
root.title("Fake News Buster")

# Create and place the components in the window
label = tk.Label(root, text="Enter a URL or a claim:")
label.pack(pady=10)

entry = tk.Entry(root, width=50)
entry.pack(pady=10)

submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()