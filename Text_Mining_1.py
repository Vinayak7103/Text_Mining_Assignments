#!/usr/bin/env python
# coding: utf-8

# # Text Mining 1

# ### TASK: NATURAL LANGUAGE PROCESSING (TEXT MINING)

#  Perform sentimental analysis on the Elon-musk tweets (Exlon-musk.csv)
# 

# IMPORTING LIBRARIES

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string # special operations on strings
import spacy # language models

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
import re
import string
import nltk


# IMPORTING DATA

# In[8]:


df=pd.read_csv("C:/Users/vinay/Downloads/Elon_musk.csv",encoding = "ISO-8859-1")


# In[9]:


pd.set_option('display.max_colwidth',200)
warnings.filterwarnings('ignore',category=DeprecationWarning)


# In[10]:


df.head()


# In[11]:


df=df.drop(['Unnamed: 0'],axis=1)


# In[12]:


df


# In[13]:


df = [Text.strip() for Text in df.Text] # remove both the leading and the trailing characters
df = [Text for Text in df if Text] # removes empty strings, because they are considered in Python as False
df[0:10]


# In[14]:


# Joining the list into one string/text
text = ' '.join(df)
text


# In[15]:


#Punctuation
no_punc_text = text.translate(str.maketrans('', '', string.punctuation)) #with arguments (x, y, z) where 'x' and 'y'
# must be equal-length strings and characters in 'x'
# are replaced by characters in 'y'. 'z'
# is a string (string.punctuation here)
no_punc_text


# In[16]:


#Tokenization
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(no_punc_text)
print(text_tokens[0:50])


# In[17]:


len(text_tokens)


# In[18]:


#Remove stopwords
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

my_stop_words = stopwords.words('english')
my_stop_words.append('the')
no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens[0:40])


# In[19]:


#Noramalize the data
lower_words = [Text.lower() for Text in no_stop_tokens]
print(lower_words[0:25])


# In[20]:


#Stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in lower_words]
print(stemmed_tokens[0:40])


# In[21]:


# NLP english language model of spacy library
nlp = spacy.load('en_core_web_sm') 


# In[22]:


# lemmas being one of them, but mostly POS, which will follow later
doc = nlp(' '.join(no_stop_tokens))
print(doc[0:40])


# In[23]:


lemmas = [token.lemma_ for token in doc]
print(lemmas[0:25])


# PERFORMING FEATURE EXTRACTION

# In[25]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(lemmas)


# In[26]:


print(vectorizer.vocabulary_)


# In[27]:


print(vectorizer.get_feature_names()[50:100])
print(X.toarray()[50:100])


# In[28]:


print(X.toarray().shape)


# PERFORMING BIGRAMS AND TRIGRAMS

# In[29]:


vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(1,3),max_features = 100)
bow_matrix_ngram =vectorizer_ngram_range.fit_transform(df)


# In[30]:


print(vectorizer_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# TF-IDF VECTORIZER

# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 500)
tf_idf_matrix_n_gram_max_features =vectorizer_n_gram_max_features.fit_transform(df)
print(vectorizer_n_gram_max_features.get_feature_names())
print(tf_idf_matrix_n_gram_max_features.toarray())


# GENERATING WORD_CLOUD

# In[32]:


# Import packages
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");


# In[33]:


# Generate wordcloud
stopwords = STOPWORDS
stopwords.add('will')
wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(text)
# Plot
plot_cloud(wordcloud)


# In[ ]:




