import re

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

def process_input(user_input):
    if is_url(user_input):
        print("Processing URL:", user_input)
        # Call function to process URL
    else:
        print("Processing claim:", user_input)
        # Call function to process claim

# Get input from the user
user_input = input("Enter a URL or a claim: ")

# Process the input
process_input(user_input)
