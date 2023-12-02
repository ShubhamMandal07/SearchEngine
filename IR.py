import os
import nltk
import numpy as np
import string


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.metrics import edit_distance




# nltk.download('punkt') 
# nltk.download('stopwords')
# nltk.download('wordnet')
#======================================================================================================================================================
Tokens = []

def tokenize_files(path):
    # Get a list of all text files in the directory
    text_files = [file for file in os.listdir(path) if file.endswith('.txt')]
    # Iterate through each text file
    for text_file in text_files:
        # print(text_file)
        file = open(path+text_file)
        text = file.read()
        # print(text)
    #     file_path = os.path.join(path, text_file)
    #     with open(file_path, 'r') as f:
    #         content = f.read()
#======================================================================================================================================================
            # Tokenize the content using NLTK's word_tokenize function
        tokens = word_tokenize(text)
#======================================================================================================================================================
            # Remove stopwords and non-alphanumeric tokens
        tokens = [token.lower() for token in tokens if
                      token.isalnum() and token.lower() not in stopwords.words('english')]
        Tokens.extend(tokens)

    return Tokens
#=======================================================================================================================================================

#lemmatizing tokens
def lemmatizing(Tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in Tokens]
#=======================================================================================================================================================
path = r'C:\Users\manda\Desktop\IR_PROGRAM\Hard_Times/'
tokens = tokenize_files(path)

final = lemmatizing(tokens)
tokens = sorted(set(final), key=final.index)
final = tokens
final = np.sort(final)
# print(final,'\n')
#=======================================================================================================================================================

# Creating inverted index with document frequency and posting list
Inverted_Index = {}
def inverted_index(Tokens,dir_path):
    
    files = os.listdir(dir_path)

    for token in Tokens:
        if token not in Inverted_Index:
            Inverted_Index[token] = {'df': 0, 'postings': []}

    for i, file in enumerate(files):
        with open(os.path.join(dir_path, file), 'r') as f:
            content = f.read()
            for j, term in enumerate(Tokens):
                if term.lower() in content.lower():
                    Inverted_Index[term]['postings'].append(i)

    for term in Tokens:
        count = len(Inverted_Index[term]['postings'])
        Inverted_Index[term]['df'] = count

    # for term, info in Inverted_Index.items():
    #     print(f"{term} : Document Frequency:{info['df']}, Posting List:{info['postings']}")
    return Inverted_Index 

# InvertedIndex = inverted_index(final,path)

#======================================================================================================================================================

def generate_bigram_index(word):
    word_with_dollars = word
    bigrams = [word_with_dollars[i:i+2] for i in range(len(word_with_dollars) - 1)]
    return bigrams

def build_bigram_index(posting_list):
    bigram_index = {}
    for word in posting_list:
        bigrams = generate_bigram_index(word.lower())
        for bigram in bigrams:
            if bigram in bigram_index:
                bigram_index[bigram].append(word)
            else:
                bigram_index[bigram] = [word]
    return bigram_index

bigram_index = build_bigram_index(final)

# print(bigram_index)
# for key,value in bigram_index.items():
#      print(key,"----->",value,"\n")

#hard ha ar rd
#======================================================================================================================================================

# def correct_spelling(query, vocabulary):
#     min_distance = float('inf')
#     corrected_term = query

#     for term in vocabulary:
#         distance = edit_distance(query, term)
#         if distance < min_distance:
#             min_distance = distance
#             corrected_term = term

#     return corrected_term

# query = input("Enter a query: ")
# corrected_query = correct_spelling(query, tokens)
# print("Corrected query:", corrected_query)


#======================================================================================================================================================

def generate_permuterm(word):
    permuterms = []
    word = word + "$"
    for i in range(len(word)):
        permuterms.append(word[i:] + word[:i])
    return permuterms


def build_permuterm_index(word_list):
    permuterm_index = {}
    for word in word_list:
        permuterms = generate_permuterm(word)
        for permuterm in permuterms:
            permuterm_index[permuterm] = word

    return permuterm_index

index = build_permuterm_index(tokens)

# print('||PERMUTERM INDEX||\n\n',index,'\n')

#======================================================================================================================================================

#soundex algorithm
def GenrateSoundexCode(word):
        consonants = {'b': 1, 'f': 1, 'p': 1, 'v': 1, 'c': 2, 'g': 2, 'j': 2, 'k': 2, 'q': 2, 's': 2, 'x': 2, 'z': 2,
                    'd': 3,
                    't': 3, 'l': 4, 'm': 5, 'n': 5, 'r': 6}

        hashCode = word
        for i in range(1, len(hashCode)):
            if hashCode[i] in consonants:
                hashCode = hashCode.replace(hashCode[i], str(consonants[word[i]]))
            else:
                hashCode = hashCode.replace(hashCode[i], " ")

        def remove(string):
            return string.replace(" ", "")

        hashCode = remove(hashCode)

        if len(hashCode) < 4:
            hashCode = hashCode + ('0' * (4 - len(hashCode)))
        else:
            hashCode = hashCode[:4]

        return hashCode

def SoundexAlgorithm(Tokens):
    SoundexCode = {}
    for index, value in enumerate(Tokens):
        key = GenrateSoundexCode(value)

        if key in SoundexCode:
            SoundexCode[key].append(value)
        else:
            SoundexCode[key] = [value]
    return SoundexCode
                        
#======================================================================================================================================================
#User Input
#======================================================================================================================================================	
# print(text_dict)
# print('===========================================================================')
# string1 = str(input('Enter string = '))
# str_soundex = create_soundex(string1)

# if str_soundex in text_dict.keys():
# 	print(f'Soundex/phonetic code of {string1} = {str_soundex} \nmatching words are = {text_dict[str_soundex]}')
    
# else:
# 	print(f'Soundex/phonetic code of {string1} = {str_soundex}')
# print('===========================================================================')        

# #inverted index        
# flag = 0
# old_info = Inverted_Index
# for i in old_info:
#         if  string1 in old_info:
#             flag=flag+1
#             a = old_info[string1]
#         else:
#               flag = 0

# if flag == 0:
#      print('This word not found')
# else:
#         print('Inverted Index of This word is:-',a)
# print('===========================================================================')
     

# generate_bigram_index(string1)
# bi = generate_bigram_index(string1)
# print(bi)
# for i in bi:
#       if i in bigram_index:
#           print('Biagram of this word',i,'----->',bigram_index[i],'\n')  
# print('===========================================================================')            
##======================================================================================================================================================
# Edit_distance

token=final

minimium = []

def Edit_Distance(s1, s2):
    # s1 = like s2 = lik
    if len(s1) > len(s2):  # len of s1 = 4 and len of s2 = 3 true
        s1, s2 = s2, s1  # s1 = lik and s2 = like

    dist = range(len(s1) + 1)  # distance = [0,1,2,3,4]

    for i2, c2 in enumerate(s2):
        dist_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                dist_.append(dist[i1])
            else:
                dist_.append(1 + min((dist[i1], dist[i1 + 1], dist_[-1])))
        dist = dist_

    return dist[-1]


def FindClosestWord(test,Tokens):
    EditDist = {}
    for s1 in Tokens:  # 0 1
        EditDist[s1] = Edit_Distance(s1, test)

    smallestElement = ""

    if EditDist[Tokens[0]] < EditDist[Tokens[1]]:
        smallestElement = Tokens[0]
        value = EditDist[Tokens[0]]
        for i in range(2, len(EditDist)):
            if EditDist[smallestElement] < EditDist[Tokens[i]]:
                smallestElement = smallestElement
                value = EditDist[smallestElement]
            else:
                smallestElement = Tokens[i]
                value = EditDist[Tokens[i]]
    else:
        smallestElement = Tokens[1]
        value = EditDist[Tokens[1]]
        for i in range(2, len(EditDist)):
            if EditDist[smallestElement] < EditDist[Tokens[i]]:
                smallestElement = smallestElement
                value = EditDist[smallestElement]
            else:
                smallestElement = Tokens[i]
                value = EditDist[smallestElement]

    return smallestElement,value

# Query = input("Enter the Query for edit distance: ")

# term,value =  FindClosestWord(Query,final)
# print(term)
# for i in token:
#     x = nltk.edit_distance([string1],i)
#     minimium.append(x)
#     # print(nltk.edit_distance(string1,i))
# print(f'minimum = {min(minimium)}')
# print('----------------------------------------------------------------------------------------')
##======================================================================================================================================================
def queryPermuterm(word):
    permuterms = []
    word = word + "$"  # Add a unique character to mark the end of the word

    for i in range(len(word)):
        permuterms.append(word[i:] + word[:i])
        wildQ = permuterms[i]
        if wildQ[-1] == "*":
            return wildQ


def getPermuterm(query,InvertedIndex):
    wildCardQuery = queryPermuterm(query)

    wildCardQuery = wildCardQuery[:len(wildCardQuery) - 1]
    print(wildCardQuery)

    querylen = len(wildCardQuery)

    documents = []
    permutermKeys = index.keys()
    flag = False
    for token in permutermKeys:
        if wildCardQuery == token[:querylen]:
            fileIndex = InvertedIndex[index[token]]
            # for i in fileIndex:
            print(index[token], "is found in ", fileIndex["postings"])
            flag = True

    if not flag:
        print("Word not Found")


# query = input("Enter The Wild Card Query :")
# getPermuterm(query)


##======================================================================================================================================================
#query processing using AND operations
# str123 = input("Enter a string for query processing: ")

# Split the string into two words using space as the delimiter
# words = str123.split()

# if len(words) == 2:
#     word1 = words[0]
#     word2 = words[1]
#     print("word1:", word1)
#     print("word2:", word2)
# else:
#     word1=words[0]

def queryprocessing(word1,word2,InvertedIndex):
    first=InvertedIndex.get(word1)
    second=InvertedIndex.get(word2)

    print(first)
    list2=[]

    for item in first['postings']:
        if item in second["postings"]:
            list2.append(item)
            print("AND operation",list2)
    return list2

# print(queryprocessing(word1,word2,InvertedIndex))

def preprocess(text):
    text = text.lower()
    text = ' '.join([char for char in text if char not in string.punctuation])
    return text

def GetRankedDocuments(doc,query):
    print("queryyyyy==",query)
    documents = []
    for i in range(len(doc)):
         with open(os.path.join("Hard_Times",doc[i]),'r') as f1:
            data1 = f1.read().lower()
            documents.append(data1)


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
        print (scores)
    # Rank Documents
    ranked_documents = [[round(score,2),doc]  for score,doc in zip(similarity_scores[0],doc)]
    # ranked_documents = [(score,doc) for score in zip(similarity_scores[0],doc)]
    ranked_documents.sort(reverse=True)
    

    # Define a threshold to determine if sentences are similar
    threshold = 0.7 

    total_r, total_nr = 0, 0
    for i in range(len(scores)):    
        if scores[i] >= threshold:
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

    Precision = round(TP/(TP + FP),2)
    print("\nPrecision = ",Precision)

    try:
        Recall = round(TP/(TP + FN),2)
        print("\nRecall = ",Recall)

        F_Measure = round(((2*Precision*Recall)/(Precision + Recall)),2)
        print("\nF-Measure = ",F_Measure,"\n")
    except :
        Recall = 0
        F_Measure = 0

    return ranked_documents, scores, Precision, Recall, F_Measure

# directory = "./Hard_Times/"
# files = [os.path.join(directory, i) for i in os.listdir(directory)]
# query = "going to rain today"
# ranked_documents, scores, Precision, Recall, F_Measure = GetRankedDocuments(files, query)


#===========================================================================================================================

# def searchs(query):
#     # Phonetic Matching using Soundex
#     soundex_code = create_soundex(query)
#     if soundex_code in text_dict:
#         phonetic_results = text_dict[soundex_code]
#     else:
#         phonetic_results = []

#     # Edit Distance
#     edit_distance_term, edit_distance_value = FindClosestWord(query, final)

#     # Query Processing using AND operations
#     query_words = query.split()
#     if len(query_words) == 2:
#         query_word1 = query_words[0]
#         query_word2 = query_words[1]
#         query_processing_results = queryprocessing(query_word1, query_word2, InvertedIndex)
#     else:
#         query_processing_results = []

#     # Prepare the results for display
#     results = f"Phonetic Matching: {phonetic_results}\n"
#     results += f"Edit Distance: Closest term: {edit_distance_term}, Edit Distance Value: {edit_distance_value}\n"
#     results += f"Query Processing (AND operation): {query_processing_results}"

#     return results



