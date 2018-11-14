import xml.etree.ElementTree as ET
import sys
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
numpy.set_printoptions(threshold=sys.maxsize)

"------------------------XML Parsing------------------------------"

#articles
tree = ET.parse("D:\\Files\\first1000.xml")
all_articles = tree.getroot()

#truths
oak_tree = ET.parse("D:\\Files\\truth1000.xml")
all_truth = oak_tree.getroot()

cachedStopWords = stopwords.words("english")

#here we are saving all those currently unused stuff like title,date,url
#and etc.
article_unused = []

#here we are saving all paragraphs ready for vectorizing
articles_ready_for_vectorizing = []

"-------------------------------------------------------------------"

"-------------------- Articles processing ------------------------- "

#going through every article in xml file
for article in all_articles:
    article_unused.append(article.attrib)
    article_text = ""
    for paragraph in article:
        if(paragraph.text != None):
            article_text = article_text+paragraph.text
            
    #removing stop words
    new_article_text = ' '.join([word for word in article_text.split() if word not in cachedStopWords])
    
    articles_ready_for_vectorizing.append(new_article_text)

#vector
vectorizer = CountVectorizer()

#matrix ready for ML models
matrix = vectorizer.fit_transform(articles_ready_for_vectorizing).todense()

with open('matrix.txt', 'w') as f:
    for vector in matrix:
        for item in vector:
            f.write("%s\n" % item)

#all vocabulary we have in articles
#print(vectorizer.vocabulary_)

"-----------------------------------------------------------------"


"-----------------------Truth processing--------------------------"

#final vector for ML testing
truth_vector = []

#going through every truth in xml file
for truth in all_truth:
    current_truth = truth.attrib
    hyperpartisan = current_truth['hyperpartisan']
    if(hyperpartisan == 'true'):
        truth_vector.append(1)
    else:
        truth_vector.append(0)

with open('truth.txt', 'w') as f:
    for item in truth_vector:
        f.write("%s\n" % item)

"------------------------------------------------------------------"
