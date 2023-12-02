# import os
# import nltk
# import numpy as np

# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.metrics import edit_distance

# # nltk.download('punkt') 
# # nltk.download('stopwords')
# # nltk.download('wordnet')
# #======================================================================================================================================================
# Tokens = []

# def tokenize_files(path):
#     # Get a list of all text files in the directory
#     text_files = [file for file in os.listdir(path) if file.endswith('.txt')]
#     # Iterate through each text file
#     for text_file in text_files:
#         # print(text_file)
#         file = open(path+text_file)
#         text = file.read()
#         # print(text)
#     #     file_path = os.path.join(path, text_file)
#     #     with open(file_path, 'r') as f:
#     #         content = f.read()
# #======================================================================================================================================================
#             # Tokenize the content using NLTK's word_tokenize function
#         tokens = word_tokenize(text)
# #======================================================================================================================================================
#             # Remove stopwords and non-alphanumeric tokens
#         tokens = [token.lower() for token in tokens if
#                       token.isalnum() and token.lower() not in stopwords.words('english')]
#         Tokens.extend(tokens)

#     return Tokens
# #=======================================================================================================================================================

# #lemmatizing tokens
# def lemmatizing(Tokens):
#     lemmatizer = WordNetLemmatizer()
#     return [lemmatizer.lemmatize(token) for token in Tokens]
# #=======================================================================================================================================================
# path = r'C:\Users\manda\Desktop\IR_PROGRAM\Hard_Times/'
# tokens = tokenize_files(path)

# final = lemmatizing(tokens)
# tokens = sorted(set(final), key=final.index)
# final = tokens
# final = np.sort(final)
# print(final,'\n')
# #=======================================================================================================================================================

# # Creating inverted index with document frequency and posting list
# Inverted_Index = {}
# def inverted_index(Tokens,dir_path,Inverted_Index):
    
#     files = os.listdir(dir_path)

#     for token in Tokens:
#         if token not in Inverted_Index:
#             Inverted_Index[token] = {'df': 0, 'postings': []}

#     for i, file in enumerate(files):
#         with open(os.path.join(dir_path, file), 'r') as f:
#             content = f.read()
#             for j, term in enumerate(Tokens):
#                 if term.lower() in content.lower():
#                     Inverted_Index[term]['postings'].append(i)

#     for term in Tokens:
#         count = len(Inverted_Index[term]['postings'])
#         Inverted_Index[term]['df'] = count

#     # for term, info in Inverted_Index.items():
#     #     print(f"{term} : Document Frequency:{info['df']}, Posting List:{info['postings']}")
#     return Inverted_Index 

# InvertedIndex = inverted_index(final,path,Inverted_Index)

# #======================================================================================================================================================

# def generate_bigram_index(word):
#     word_with_dollars = word
#     bigrams = [word_with_dollars[i:i+2] for i in range(len(word_with_dollars) - 1)]
#     return bigrams

# def build_bigram_index(posting_list):
#     bigram_index = {}
#     for word in posting_list:
#         bigrams = generate_bigram_index(word.lower())
#         for bigram in bigrams:
#             if bigram in bigram_index:
#                 bigram_index[bigram].append(word)
#             else:
#                 bigram_index[bigram] = [word]
#     return bigram_index

# bigram_index = build_bigram_index(final)

# # print(bigram_index)
# # for key,value in bigram_index.items():
# #      print(key,"----->",value,"\n")

# #hard ha ar rd
# #======================================================================================================================================================

# # def correct_spelling(query, vocabulary):
# #     min_distance = float('inf')
# #     corrected_term = query

# #     for term in vocabulary:
# #         distance = edit_distance(query, term)
# #         if distance < min_distance:
# #             min_distance = distance
# #             corrected_term = term

# #     return corrected_term

# # query = input("Enter a query: ")
# # corrected_query = correct_spelling(query, tokens)
# # print("Corrected query:", corrected_query)


# #======================================================================================================================================================

# def generate_permuterm(word):
#     permuterms = []
#     word = word + "$"
#     for i in range(len(word)):
#         permuterms.append(word[i:] + word[:i])
#     return permuterms


# def build_permuterm_index(word_list):
#     permuterm_index = {}
#     for word in word_list:
#         permuterms = generate_permuterm(word)
#         for permuterm in permuterms:
#             permuterm_index[permuterm] = word

#     return permuterm_index

# index = build_permuterm_index(tokens)

# print('||PERMUTERM INDEX||\n\n',index,'\n')

# #======================================================================================================================================================

# #soundex algorithm
# def soundex(name):
# 	name = name.upper()
# 	soundex = ""
# 	soundex += name[0]
# 	dictionary = {"BFPV": "1", "CGJKQSXZ":"2", "DT":"3", "L":"4", "MN":"5", "R":"6", "AEIOUHWY":"0"}

# 	for char in name[1:]:
# 		for key in dictionary.keys():
# 			if char in key:
# 				code = dictionary[key]
# 				if code != soundex[-1]:
# 					soundex += code
# 					# print(soundex)

# 	soundex = soundex.replace("0", "")
# 	soundex = soundex[:4].ljust(4, "0")
# 	return soundex

# # create_soundex('bangalore')

# text_dict = { }

# data = Tokens
	
# for i in data:
# 		x = soundex(i)
# 		if x in text_dict:
# 			if i not in text_dict[x]:
# 				text_dict[x].append(i)
# 		else:
# 			text_dict[x] = [i]
                        
# #======================================================================================================================================================
# #User Input
# #======================================================================================================================================================	
# # print(text_dict)
# print('===========================================================================')
# string1 = str(input('Enter string = '))
# str_soundex = soundex(string1)

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
# # print(bi)
# for i in bi:
#       if i in bigram_index:
#           print('Biagram of this word',i,'----->',bigram_index[i],'\n')  
# print('===========================================================================')            
# ##======================================================================================================================================================
# # Edit_distance


# # for i in final:
# #      if i == 'x':
# #           print('exist')
# #           quit()



# # def Edit_Distance(s1, s2):
# #     # s1 = like s2 = lik
# #     if len(s1) > len(s2):  # len of s1 = 4 and len of s2 = 3 true
# #         s1, s2 = s2, s1  # s1 = lik and s2 = like

# #     dist = range(len(s1) + 1)  # distance = [0,1,2,3,4]

# #     for i2, c2 in enumerate(s2):
# #         dist_ = [i2 + 1]
# #         for i1, c1 in enumerate(s1):
# #             if c1 == c2:
# #                 dist_.append(dist[i1])
# #             else:
# #                 dist_.append(1 + min((dist[i1], dist[i1 + 1], dist_[-1])))
# #         dist = dist_

# #     return dist[-1]


# # def FindClosestWord(test,Tokens):
# #     EditDist = {}
# #     for s1 in Tokens:  # 0 1
# #         EditDist[s1] = Edit_Distance(s1, test)

# #     smallestElement = ""

# #     if EditDist[Tokens[0]] < EditDist[Tokens[1]]:
# #         smallestElement = Tokens[0]
# #         value = EditDist[Tokens[0]]
# #         for i in range(2, len(EditDist)):
# #             if EditDist[smallestElement] < EditDist[Tokens[i]]:
# #                 smallestElement = smallestElement
# #                 value = EditDist[smallestElement]
# #             else:
# #                 smallestElement = Tokens[i]
# #                 value = EditDist[Tokens[i]]
# #     else:
# #         smallestElement = Tokens[1]
# #         value = EditDist[Tokens[1]]
# #         for i in range(2, len(EditDist)):
# #             if EditDist[smallestElement] < EditDist[Tokens[i]]:
# #                 smallestElement = smallestElement
# #                 value = EditDist[smallestElement]
# #             else:
# #                 smallestElement = Tokens[i]
# #                 value = EditDist[smallestElement]

# #     return smallestElement,value

# # Query = input("Enter the Query for edit distance: ")

# # term,value =  FindClosestWord(Query,final)
# # print(term)

# new_tokens = []
# for i in final:
#      if len(i) >3:
#           new_tokens.append(i)
          
# token=new_tokens

# minimium = []
# for i in token:
#     x = nltk.edit_distance([string1],i)
#     minimium.append([x,i])
#     # print(nltk.edit_distance(string1,i))
# sorted_minimum = sorted(minimium)
# print('Edit Distance',sorted_minimum[:6])
# print('----------------------------------------------------------------------------------------')
# ##======================================================================================================================================================
# def queryPermuterm(word):
#     permuterms = []
#     word = word + "$"  # Add a unique character to mark the end of the word

#     for i in range(len(word)):
#         permuterms.append(word[i:] + word[:i])
#         wildQ = permuterms[i]
#         if wildQ[-1] == "*":
#             return wildQ


# def getPermuterm(query):
#     wildCardQuery = queryPermuterm(query)

#     wildCardQuery = wildCardQuery[:len(wildCardQuery) - 1]
#     print(wildCardQuery)

#     querylen = len(wildCardQuery)

#     documents = []
#     permutermKeys = index.keys()
#     flag = False
#     for token in permutermKeys:
#         if wildCardQuery == token[:querylen]:
#             fileIndex = InvertedIndex[index[token]]
#             # for i in fileIndex:
#             print(index[token], "is found in ", fileIndex["postings"])
#             flag = True

#     if not flag:
#         print("Word not Found")


# query = input("Enter The Wild Card Query :")
# getPermuterm(query)


# ##======================================================================================================================================================
# #query processing using AND operations
# str123 = input("Enter a string for query processing: ")

# # Split the string into two words using space as the delimiter
# words = str123.split()

# if len(words) == 2:
#     word1 = words[0]
#     word2 = words[1]
#     print("word1:", word1)
#     print("word2:", word2)
# else:
#     word1=words[0]

# def queryprocessing(word1,word2,InvertedIndex):
#     first=InvertedIndex.get(word1)
#     second=InvertedIndex.get(word2)

#     print(first)
#     list2=[]

#     for item in first['postings']:
#         if item in second["postings"]:
#             list2.append(item)
#             print("AND operation",list2)
#     return list2

# print(queryprocessing(word1,word2,InvertedIndex))
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

location = os.listdir('Hard_Times/')
retrieved_documents = {}

for i in location:
    path = 'Hard_Times/' + i 
    with open(path , 'r', encoding='utf-8') as file:
        data = file.read().lower()
        data = data.replace('\n', ' ').replace('\t', ' ')
        retrieved_documents[i] = data

# query = "Louis heard the click"
query = "Louis glanced"

extras = ['!', '@', '#', '$', "%", "^", "&" ,"*", "(",")", "<",">", ",", ".", "[","]", "{","}", "_"]

if any(char in query for char in extras):
    for char in extras:
        query = query.replace(char, '')

all_retrieved_documents_names = []
all_retrieved_documents_values= []


retrieved_documents_list = []
# relevant_documents_list = []

query_words = query.lower().split()

for key, value in retrieved_documents.items():

    document_words = value

    #check if any extras is/are in document_words
    if any(char in document_words for char in extras):
        for char in extras:
            document_words = document_words.replace(char, '')
    document_words=document_words.lower().split(" ")

    if all(word in document_words for word in query_words):
        all_retrieved_documents_names.append(key)
        # all_retrieved_documents_values.append(value)

        retrieved_documents_list.append(key)
        
# print(all_retrieved_documents_names)
relevant_documents_list = ['Chapter 11.txt', 'Chapter 13.txt','Chapter 14.txt']   #Hardcorded ---- but user should select which documents are relevent to him/her and that should be recorded


def calculate_cosine_similarity(query_vector, document_vectors):
    similarity_scores = cosine_similarity(query_vector, document_vectors)
    return similarity_scores

def evaluate_search_results(query, retrieved_documents, relevant_documents):
    # Vectorize the query and documents using TF-IDF
    vectorizer = TfidfVectorizer()
    vectorizer.fit(retrieved_documents + [query])  # Include the query for vectorization
    query_vector = vectorizer.transform([query])
    document_vectors = vectorizer.transform(retrieved_documents)

    # Calculate cosine similarity
    similarity_scores = calculate_cosine_similarity(query_vector, document_vectors)

    # Rank documents based on similarity scores
    ranked_results = [doc for _, doc in sorted(zip(similarity_scores[0], retrieved_documents), reverse=True)]

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = len(set(ranked_results) & set(relevant_documents))
    fp = len(set(ranked_results) - set(relevant_documents))
    fn = len(set(relevant_documents) - set(ranked_results))

    # Calculate Precision, Recall, F-measure
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return ranked_results, precision, recall, f_measure


print('all_retrieved_documents_names = ',all_retrieved_documents_names, '\nrelevant_documents_list =',relevant_documents_list)
ranked_results, precision, recall, f_measure = evaluate_search_results(query, all_retrieved_documents_names, relevant_documents_list)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-measure: {f_measure}")