import pandas as pd
from subprocess import call
import re
import tkinter as tk
from tkinter import messagebox
from tkinter import font
from tkinter import ttk
from PIL import Image, ImageTk  # Use PIL for image handling
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
    return result, response, final_score

def on_submit():
    userInput = entry.get()
    result, response, final_score = process_input(userInput)
    
    # Displaying the output in bold along with the final score
    output_text = f"{result}\n\nFinal Score: {final_score:.16f}"
    output_text_widget.config(state=tk.NORMAL)
    output_text_widget.delete(1.0, tk.END)
    output_text_widget.insert(tk.END, output_text)
    output_text_widget.tag_configure("bold", font=("Helvetica", 12, "bold"))
    output_text_widget.tag_add("bold", "1.0", "2.0")
    output_text_widget.config(state=tk.DISABLED)
    
    # Displaying the response data in a tabular form
    table_data = [(entry['BodyID'], entry['Stances'], entry['SourceName'], entry['URL']) for entry in response]
    table_columns = ["Body ID", "Stances", "Source Name", "URL"]
    response_table.delete(*response_table.get_children())
    for row in table_data:
        response_table.insert('', 'end', values=row)
    
    # Make URL column clickable
    response_table.tag_configure('url', foreground='blue', underline=True)
    response_table.bind('<ButtonRelease-1>', on_url_click)

def on_url_click(event):
    item = response_table.selection()[0]
    url = response_table.item(item, 'values')[3]
    messagebox.showinfo("URL", url)

# Initialize Tkinter
root = tk.Tk()
root.title("Fake Bananas")
root.geometry("800x600")
root.configure(bg='#f0f0f0')

# Set font styles
title_font = font.Font(family="Helvetica", size=16, weight="bold")
label_font = font.Font(family="Helvetica", size=12)
entry_font = font.Font(family="Helvetica", size=12)
button_font = font.Font(family="Helvetica", size=12, weight="bold")
text_font = font.Font(family="Helvetica", size=14, weight="bold")

# Load the image
image_path = "C:/Users/91846/OneDrive/Desktop/EDAI_2/logo.png"  # Change this to your image path
image = Image.open(image_path)
image = image.resize((50, 50), Image.LANCZOS)
photo = ImageTk.PhotoImage(image)

table_columns = ["Body ID","Stances", "Source Name","URL"]

# Create and place the components in the window
frame = tk.Frame(root, bg='#f0f0f0')
frame.pack(pady=20)

image_label = tk.Label(frame, image=photo, bg='#f0f0f0')
image_label.pack(side="left", padx=10)

title_label = tk.Label(frame, text="Fake Bananas", font=title_font, bg='#f0f0f0', fg="black")
title_label.pack(side="left")

# Add additional text
additional_text = tk.Label(root, text="Check your facts before you slip on them.\nValidate your article claims against our machine learning system to predict its credibility", 
                            font=text_font, bg='#f0f0f0', wraplength=700, justify="center")
additional_text.pack(pady=10)

label = tk.Label(root, text="Enter a URL or a claim:", font=label_font, bg='#f0f0f0')
label.pack(pady=10)

entry = tk.Entry(root, width=50, font=entry_font)
entry.pack(pady=10)

submit_button = tk.Button(root, text="Submit", font=button_font, bg='#007BFF', fg='white', command=on_submit)
submit_button.pack(pady=10)

# Display output text in a Text widget
output_text_widget = tk.Text(root, height=6, width=70, wrap=tk.WORD, font=("Helvetica", 12), bg='#f0f0f0')
output_text_widget.pack(pady=10)
output_text_widget.insert(tk.END, "Output\n", "bold")
output_text_widget.insert(tk.END, "Final Score: ", "bold")

# Create a frame for the response data table
table_frame = tk.Frame(root, bg='#f0f0f0')
table_frame.pack(padx=10, pady=10, fill='both', expand=True)

# Create a Treeview widget for the response data table
response_table = ttk.Treeview(table_frame, columns=("Body ID", "Stances", "Source Name", "URL"), show='headings')
response_table.pack(side=tk.LEFT, fill='both', expand=True)

# Add a scrollbar to the Treeview widget
table_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=response_table.yview)
table_scroll.pack(side=tk.RIGHT, fill='y')
response_table.configure(yscrollcommand=table_scroll.set)

# Insert response data into the Treeview widget
for col in ["Body ID", "Stances", "Source Name", "URL"]:
    response_table.heading(col, text=col, anchor=tk.CENTER)
    response_table.tag_configure(col, font=("Helvetica", 12, "bold"))

# Run the Tkinter event loop
root.mainloop()