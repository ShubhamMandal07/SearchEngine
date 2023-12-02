from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import os

def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

def GetRankedDocument(doc,query):
    documents = []
    for i in range(len(doc)):
        with open(os.path.join("Hard_Times",doc[i]),'r') as f1:
            data1 = f1.read().lower()
            documents.append(data1)

    # Create a DataFrame for IDF values
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    idf_values = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    preprocessed_query = preprocess(query)
    query_tfidf = tfidf_vectorizer.transform([preprocessed_query])

    # Calculate Similarity
    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix)
    scores = []
    for i in range(len(similarity_scores[0])):
        scores.append(round(similarity_scores[0][i],2))
        print(scores)

    # Rank Documents
    ranked_documents = [[round(score,2),doc]  for score,doc in zip(similarity_scores[0],doc)]
    #ranked_documents = [(score,doc) for score in zip(similarity_scores[0],doc)]
    print(ranked_documents)
    ranked_documents.sort(reverse=True)
   

# directory = "Hard_Times"
# # files = [os.path.join(directory, i) for i in os.listdir(directory)]
# files = [i for i in os.listdir(directory)]
# query = "plain"
# #query = "going to rain"
# ranked_documents, scores = GetRankedDocuments(files, query)

# Define a threshold to determine if sentences are similar
    threshold = 0.5
    threshold = max(scores) if max(scores) < threshold else threshold

    print(threshold)

    total_r, total_nr = 0, 0
    for i in range(len(scores)):    
        if ranked_documents[i][0] >= threshold:
            total_r = total_r + 1
            print("\n",ranked_documents[i]," : relevant")
        else:
            total_nr = total_nr + 1
            print("\n",ranked_documents[i]," : non-relevant")

    Total = total_r + total_nr
    print("\nTotal Documents = ",Total)
    print("Total Docs that are Relevant = ",total_r)
    print("Total Docs that are Non-Relevant = ",total_nr)

    TP = total_r   # True Positive    
    FP = total_nr  # False Positive
    FN = total_r - TP

    print("\nTP = ",TP)
    print("FP = ",FP)
    print("FN = ",FN)

    if ((TP + FP) == 0):
        Precision = 0.0
    else:
        Precision = round(TP/(TP + FP),2)   
    print("\nPrecision = ",Precision)

    if ((TP + FN) == 0):
        Recall = 0.0
    else:
        Recall = round(TP/(TP + FN),2)
    
    print("\nRecall = ",Recall)

    if ((Precision + Recall) == 0):
        F_Measure = 0.0
    else:
        F_Measure = round(((2*Precision*Recall)/(Precision + Recall)),2)
    print("\nF-Measure = ",F_Measure,"\n")
    return ranked_documents, scores , Precision,Recall,F_Measure