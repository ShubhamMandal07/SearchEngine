import eel
from IR import *
from PRF import GetRankedDocument
eel.init("web")

path = r'C:\Users\manda\Desktop\project to show\SearchEngine\Hard_Times/'
text_files = [file for file in os.listdir(path) if file.endswith('.txt')]
documents = []
for text_file in text_files:
        file_path = os.path.join(path, text_file)
        with open(file_path, 'r') as f:
            documents.append(f.name.rsplit('/',1)[-1])
tokens = tokenize_files(path)

final = lemmatizing(tokens)
Tokens = sorted(set(final), key=final.index)
final = Tokens


InvertedIndex = inverted_index(final,path)
bigram_index = build_bigram_index(Tokens)
index = build_permuterm_index(Tokens)
Soundex = SoundexAlgorithm(Tokens)
# print(Soundex)
# str_SoundexAlgorithm
@eel.expose
def run_Inverted():
    output_data = []
    for token in Tokens:
        if token in InvertedIndex:
            token_data = {
                "token": token,
                "frequency": InvertedIndex[token]['df'],
                "posting_list": InvertedIndex[token]['postings']
            }
            output_data.append(token_data)
    return output_data

@eel.expose
def run_Biagram():
    output_data = []
    for key, value in bigram_index.items():
        token_data ={
            "token": key,
            "frequency": value,
        }
        output_data.append(token_data)
    return output_data

@eel.expose
def run_Permuterm():
    output_data = []
    for key, value in index.items():
        token_data ={
            "token": key,
            "frequency": value,
        }
        output_data.append(token_data)
    return output_data

@eel.expose
def run_Soundex():
    output_data = []
    for key, value in Soundex.items():
        token_data ={
            "token": key,
            "frequency": value,
        }
        output_data.append(token_data)
    return output_data

@eel.expose
def run_multiple_query(query):
    ranked_documents, scores, Precision, Recall, F_Measure= GetRankedDocument(documents,query)

    search_results = {
        "token": query, 
        "scores": scores,
        "ranked_documents": ranked_documents[:6],
        "document_ranking": ranked_documents[:6],
        "Precision": Precision,
        "Recall": Recall,
        "F_Measure": F_Measure,
        # "ranked_documents": ranked_documents
                # "RankedDoc":RankedDocuments
    }
    return search_results


@eel.expose
def search(query):
    docName = []
    matchingSoundex = []
    smallestElement,value = FindClosestWord(query,Tokens)
    HashCode = GenrateSoundexCode(smallestElement)
    print(HashCode)
    Postinglist = InvertedIndex.get(smallestElement)

    for code,value in Soundex.items():
        if HashCode == code:
            matchingSoundex = value

    for i in Postinglist['postings']:
        docName.append(documents[i])

        print(docName)
    ranked_documents, scores, Precision, Recall, F_Measure= GetRankedDocument(documents,query)

    search_results = {
        "token": smallestElement, 
        "frequency": HashCode,
        "matchingCode": matchingSoundex,
        "posting_list": Postinglist['postings'],
        "scores": scores,
        "Doc_names": docName,
        "Precision": Precision,
        "Recall": Recall,
        "F_Measure": F_Measure,
        "ranked_documents": ranked_documents[:6]
                # "RankedDoc":RankedDocuments
    }
    return search_results


if __name__ == "__main__":
# Run the GUI
    eel.start("IR.html",mode="default")

