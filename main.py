import nltk
import re
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Sample input text (replace this with your actual text)
text = """
Natural Language Processing (NLP) enables machines to understand human language. 
NLP tasks include text classification, sentiment analysis, machine translation, and more.
"""

# Step 1: Lowercase
text = text.lower()

# Step 2: Remove punctuation and non-alphabetic characters
text = re.sub(r'[^a-z\s]', '', text)

# Step 3: Tokenization
tokens = word_tokenize(text)

# Step 4: Remove stop words
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]

# Step 5: POS tagging for lemmatization
pos_tags = pos_tag(tokens)

# Step 6: Mapping NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# Step 7: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]

# Step 8: Join tokens into a single string for the word cloud
processed_text = ' '.join(lemmatized_tokens)

# Step 9: Generate and display the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud After NLP Preprocessing", fontsize=16)
plt.show()