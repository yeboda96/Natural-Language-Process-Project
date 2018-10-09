
# coding: utf-8

# # Document Clustering and Topic Modeling
In this project, I use unsupervised learning models to cluster unlabeled documents into different groups, visualize the results，identify their latent topics/structures and sentiment analysis. Moreover, I compared the difference between k-means and LDA, in terms of rating and sentiment analysis
# ## Contents

# <ul>
# <li>[Part 0: Translate Reviews](#Part-0:-Translate-Reviews)
# <li>[Part 1: Load Data](#Part-1:-Load-Data)
# <li>[Part 2: Tokenizing and Stemming](#Part-2:-Tokenizing-and-Stemming)
# <li>[Part 3: TF-IDF](#Part-3:-TF-IDF)
# <li>[Part 4: K-means clustering](#Part-4:-K-means-clustering)
# <li>[Part 5: Topic Modeling - Latent Dirichlet Allocation](#Part-5:-Topic-Modeling---Latent-Dirichlet-Allocation)
# <li>[Part 6: Sentiment Analysis](#Part-6:-Sentiment-Analysis)
# <li>[Part 7: Comparison K-means and LDA](#Part-7:-Comparison-K-Means-and-LDA)
# </ul>

# # Part 0: Translate Reviews

# In[ ]:


from google.cloud import translate
import os
import csv
import time
import pandas as pd

#setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/yebod/Downloads/My Project 61912-be63adc2c36e.json"
li_summary=[]
li_review=[]

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def translate_text(text,target='en'):
    print("text is"+text)
    li_original=chunkIt(text,len(text)/800)
    output="";
    print (li_original)


    for i in li_original:
        translate_client = translate.Client()
        print ("i is"+i)
        result = translate_client.translate(i, target_language=target)
        output=output+result['translatedText']
        print ("output is "+ output)
        time.sleep(3)
    return output

with open("review subsample.csv",encoding="utf-8") as csvinput:

    reader = csv.reader(csvinput)
    for row in reader:
        summary_text=row[1]
        li_summary.append(translate_text(summary_text))
        if(row[2]==None):
            li_review.append("")
        else:
            review_text=row[2]
            li_review.append(translate_text(review_text))
        time.sleep(1)

    df = pd.DataFrame(li_review, columns=["review"])
    df.to_csv('wonnai_review.csv',index=False)
    df2=pd.DataFrame(li_summary,columns=["summary"])
    df2.to_csv('wongnai_summary.csv',index=False)


# # Part 1: Load Data

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
import re
import os

from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import lda


# Read data from files. In summary, we have about 100 thousand user reviews

# In[2]:


userReviews=pd.read_csv('../data/translation.txt',sep= '\t',error_bad_lines=False,header=None,usecols=[0,1,2],nrows=5000)
#remove non ascii
userReviews.replace({u'[^\x00-\x7F]+':' '}, regex=True, inplace=True)
userReviews=userReviews.loc[userReviews[2]!=' ']
userReviews= userReviews[userReviews[2].notnull()]
reviews=userReviews[2]


# In[3]:


reviews[:10]


# In[4]:


#remove Non Ascii

'''
asciiReviews=[]
for line in reviews:
    line=str(line).strip().decode("ascii","ignore").encode("ascii")
    if line=="":continue
    asciiReviews.append(line)
asciiReviews[0:5]


'''

        


# # Part 2: Tokenizing and Stemming

# Load stopwords and stemmer function from NLTK library.
# Stop words are words like "a", "the", or "in" which don't convey significant meaning.
# Stemming is the process of breaking a word down into its root.

# In[5]:


# Use nltk's English stopwords.
stopwords = nltk.corpus.stopwords.words('english')

print "We use " + str(len(stopwords)) + " stop-words from nltk library."
print stopwords[:10]


# In[6]:


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenization_and_stemming(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stopwords]
#     tokens=[]
#     for sent in nltk.sent_tokenize(text):
#         for word in nltk.word_tokenize(sent):
#             if word not in stopwords:
#                 tokens.append(word);    
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)

    for token in tokens:
        if re.search('[a-zA-Z]', token) and '&' not in token and len(token)>1:
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenization(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stopwords]
    filtered_tokens = []
    for token in tokens:
        #word should be composed by alphabets, no use '&', and everything is Ascii.
        if re.search('[a-zA-Z]', token) and '&' not in token and len(token)>1:
            filtered_tokens.append(token)
    return filtered_tokens


# In[7]:


tokenization_and_stemming("she looked at her father's arm.")


# Use our defined functions to analyze (i.e. tokenize, stem) our synoposes.

# In[8]:


docs_stemmed = []
docs_tokenized = []
for i in reviews:
    try:
        tokenized_and_stemmed_results = tokenization_and_stemming(i)
        docs_stemmed.extend(tokenized_and_stemmed_results)

        tokenized_results = tokenization(i)
        docs_tokenized.extend(tokenized_results)
    except:
        print i


# In[9]:


docs_tokenized[:10]


# In[10]:


docs_stemmed[:10]


# Create a mapping from stemmed words to original tokenized words for result interpretation.

# In[11]:


vocab_frame_dict = {docs_stemmed[x]:docs_tokenized[x] for x in range(len(docs_stemmed))}
print vocab_frame_dict['delici']


# # Part 3: TF-IDF

# In[12]:


#define vectorizer parameters
tfidf_model = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.1, stop_words='english',
                                 use_idf=True, tokenizer=tokenization_and_stemming, ngram_range=(1,1))

tfidf_matrix = tfidf_model.fit_transform(reviews) #fit the vectorizer to synopses

print "In total, there are " + str(tfidf_matrix.shape[0]) +       " summaries and " + str(tfidf_matrix.shape[1]) + " terms."


# In[13]:


tfidf_matrix[0]


# In[14]:


tfidf_model.get_params()


# Save the terms identified by TF-IDF.

# In[15]:


tf_selected_words = tfidf_model.get_feature_names()


# # (Optional) Calculate Document Similarity

# In[16]:


from sklearn.metrics.pairwise import cosine_similarity
cos_matrix = cosine_similarity(tfidf_matrix)
print cos_matrix


# In[17]:


cos_matrix


# # Part 4: K-means clustering

# In[18]:


from sklearn.cluster import KMeans

num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()


# In[19]:


clusters[0:10]


# In[20]:


userReviews['K-Means']=clusters


# In[21]:


userReviews[-10:]


# ## 4.1. Analyze K-means Result

# In[22]:


# create DataFrame films from all of the input files.
frame = pd.DataFrame(clusters , columns = ['cluster'])


# In[23]:


print "Number of films included in each cluster:"
frame['cluster'].value_counts().to_frame()


# In[24]:


print "<Document clustering result by K-means>"

#km.cluster_centers_ denotes the importances of each items in centroid.
#We need to sort it in decreasing-order and get the top k items.
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

Cluster_keywords_summary = {}
for i in range(num_clusters):
    print "Cluster " + str(i) + " words:" ,
    Cluster_keywords_summary[i] = []
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        Cluster_keywords_summary[i].append(vocab_frame_dict[tf_selected_words[ind]])
        print vocab_frame_dict[tf_selected_words[ind]] + ",",
    print
    
    
    


# ## 4.2. Plot K-means Result

# In[25]:


pca = decomposition.PCA(n_components=2)
tfidf_matrix_np=tfidf_matrix.toarray()
pca.fit(tfidf_matrix_np)
X = pca.transform(tfidf_matrix_np)

xs, ys = X[:, 0], X[:, 1]

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
#set up cluster names using a dict
cluster_names = {}
for i in range(num_clusters):
    cluster_names[i] = ", ".join(Cluster_keywords_summary[i])


# In[26]:


get_ipython().magic(u'matplotlib inline')

#create data frame with PCA cluster results
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters)) 
groups = df.groupby(clusters)

# set up plot
fig, ax = plt.subplots(figsize=(16, 9))
#Set color for each cluster/group
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')

ax.legend(numpoints=1,loc=4)  #show legend with only 1 point, position is right bottom.

plt.show() #show the plot


# In[27]:


plt.close()


# # Part 5: Topic Modeling - Latent Dirichlet Allocation

# In[28]:


from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=5, learning_method = 'online')
tfidf_matrix_lda = (tfidf_matrix * 100)
tfidf_matrix_lda = tfidf_matrix_lda.astype(int)


# In[29]:


lda.fit(tfidf_matrix_lda)


# In[30]:


#5 group, 44 selected words
topic_word = lda.components_
print topic_word.shape


# In[31]:


n_top_words = 10
topic_keywords_list = []
for i, topic_dist in enumerate(topic_word):
    #Here we select top(n_top_words-1)
    lda_topic_words = np.array(tf_selected_words)[np.argsort(topic_dist)][:-n_top_words:-1] 
    for j in range(len(lda_topic_words)):
        lda_topic_words[j] = vocab_frame_dict[lda_topic_words[j]]
    topic_keywords_list.append(lda_topic_words.tolist())


# <li> "model.topic_word_" saves the importance of tf_selected_words in LDA model, i.e. words similarity matrix
# <li> The shape of "model.topic_word_" is (n_topics,num_of_selected_words)
# <li> "model.doc_topic_" saves the document topic results, i.e. document topic matrix.
# <li> The shape of "model.doc_topic_" is (num_of_documents, n_topics)

# In[32]:


#3788 docs, and 5 topics
doc_topic = lda.transform(tfidf_matrix_lda)
print doc_topic.shape


# In[33]:


for i in range(5):
    print "Cluster " + str(i) + " words: " + ", ".join(topic_keywords_list[i])


# In[34]:


list=[i.argmax() for i in doc_topic]


# In[35]:


userReviews['LDA']=list
userReviews['topic1_score']=[i[0] for i in doc_topic]
userReviews['topic2_score']=[i[1] for i in doc_topic]
userReviews['topic3_score']=[i[2] for i in doc_topic]
userReviews['topic4_score']=[i[3] for i in doc_topic]
userReviews['topic5_score']=[i[4] for i in doc_topic]


# In[36]:


userReviews[0:10]


# # part6: Sentiment Analysis

# In[37]:


#load data from official doc
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
n_instances = 100
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
len(subj_docs), len(obj_docs)


# In[38]:


#mark negation
train_subj_docs = subj_docs[:80]
test_subj_docs = subj_docs[80:100]
train_obj_docs = obj_docs[:80]
test_obj_docs = obj_docs[80:100]
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs

sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])


# In[39]:


#unigram feats
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)



# In[40]:


unigram_feats[:10]


# In[41]:


#train and test
test_set = sentim_analyzer.apply_features(testing_docs)
training_set = sentim_analyzer.apply_features(training_docs)

trainer=NaiveBayesClassifier.train
classifier=sentim_analyzer.train(trainer,training_set)

results = sentim_analyzer.evaluate(test_set)

#for key ,value in sorted(sentim_analyzer.evaluate(test_set).items()):
#    print('{0}:{1}'.format(key,value))


# In[42]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
neg,neu,pos,compound=[],[],[],[]
for sentence in userReviews[2]:
    ss = sid.polarity_scores(str(sentence))
    neg.append(ss['neg'])
    neu.append(ss['neu'])
    pos.append(ss['pos'])
    compound.append(ss['compound'])
    #print("{:-<40} {}".format(sentence, str(ss)))
    #print()


# In[43]:


neg[0:10]


# In[44]:


userReviews['neg']=neg
userReviews['neu']=neu
userReviews['pos']=pos
userReviews['compound']=compound


# In[45]:


userReviews[-10:]


# # Part 7: Comparison between K-Means and LDA

# In[46]:


#sentiment Analysis in terms of K-means group
userReviews.groupby('K-Means', as_index=False)['neg','neu','pos','compound'].mean()


# In[47]:


#sentiment Analysis in terms of LDA
userReviews.groupby('LDA',as_index=False)['neg','neu','pos','compound'].mean()


# # Appendix: K-means

# In[48]:


from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], s=50);


# In[49]:


from sklearn.cluster import KMeans
est = KMeans(4)  # 4 clusters
est.fit(X)
y_kmeans = est.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50);

