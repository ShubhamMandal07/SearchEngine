# SearchEngine
This is the Information Reterival project we have done Using Various Algrothim .

 CORPUS : Charles Dickens’s Hard Times Novel
 Total Documents in Corpus : 10


## Explanation of Process:
 Technology used for logic: Python
 Technology used for GUI: Eel, Html, Css, JavaScript

  Corpus Selection: Read the data from the corpus.
  Data Cleansing Process: Used nltk library of python for
Tokenisation, Stopwords removal and Lemmatization.
  Store those words in Dataset for further usage.
  Then find the Inverted index, Permuterm index, Soundex code
and Bigram index from Dataset.
  To search General query:
o The user will search for any query, if that term of the
query is present in any of the document, then those
documents will be displayed.
o The user also has an option to read the text documents if
the user clicks on them. 
o The value of Precision, Recall And F-Measure will also
get calculated accordingly.


  To search Wildcard query:
o The user will search for a wildcard query, if that word is
present in Permuterm index then the word from the
Permuterm index will be displayed to the user. 
o Then that word will be searched within each of the
documents and those documents will be displayed along
with their Precision, Recall and F-measure value. 
o The user also has an option to read the text documents if
the user clicks on them. 

  For misspelled words:
o If a user enters a misspelled word that is not present in
the novel’s chapter, then using Edit distance the user will
be suggested the correct word and will display the
documents which contains that word along with its
Precision, Recall and F-measure.

 It also contains a ‘More’ button which displays the Soundex
Code, Inverted Index, Bigram Index and Permuterm Index of
the whole words which are present in every documents.

## ss and gui
 Query:
 Displays the documents which contains the word alongwith its Precision, Recall and F-measure value:
 Displaying documents to read:
