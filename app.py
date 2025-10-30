

from flask import Flask, render_template, request
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from PyPDF2 import PdfFileReader

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        pdf_file = request.files.get('pdf_input')
        percentage = float(request.form['percentage'])

        if pdf_file:
            pdf_content = extract_text_from_pdf(pdf_file)
            final_summary = text_summarizer(pdf_content, percentage)
            return render_template('index.html', pdf_content=pdf_content, final_summary=final_summary)
        else:
            final_summary = text_summarizer(user_input, percentage)
            return render_template('index.html', user_input=user_input, final_summary=final_summary)

    return render_template('index.html')

def text_summarizer(text, percentage):
    nlp = spacy.load('en_core_web_sm')

    # pass the text into the nlp function
    doc = nlp(text)

    ## The score of each word is kept in a frequency table
    tokens = [token.text for token in doc]
    freq_of_word = dict()

    # Text cleaning and vectorization
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in freq_of_word.keys():
                    freq_of_word[word.text] = 1
                else:
                    freq_of_word[word.text] += 1

    # Maximum frequency of word
    max_freq = max(freq_of_word.values())

    # Normalization of word frequency
    for word in freq_of_word.keys():
        freq_of_word[word] = freq_of_word[word] / max_freq

    # In this part, each sentence is weighed based on how often it contains the token.
    sent_tokens = [sent for sent in doc.sents]
    sent_scores = dict()
    for sent in sent_tokens:
        for word in sent:
            if word.text.lower() in freq_of_word.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = freq_of_word[word.text.lower()]
                else:
                    sent_scores[sent] += freq_of_word[word.text.lower()]

    len_tokens = int(len(sent_tokens) * percentage)

    # Summary for the sentences with maximum score. Here, each sentence in the list is of spacy.span type
    summary = nlargest(n=len_tokens, iterable=sent_scores, key=sent_scores.get)

    # Prepare for final summary
    final_summary = [word.text for word in summary]

    # convert to a string
    summary = " ".join(final_summary)

    # Return final summary
    return summary
    

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfFileReader(pdf_file)
    text = ""
    for page_num in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page_num).extractText()
    return text

if __name__ == '__main__':
    app.run(debug=True)
