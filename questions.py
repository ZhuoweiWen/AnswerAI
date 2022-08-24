import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    res = dict()
    for filename in os.listdir(directory):
        #Check if it is a txt file
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), encoding="utf8") as file:           
                res[filename] = file.read()

    return res

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = nltk.word_tokenize(document.lower())

    #Filter out punctuation and stopwords
    for t in tokens.copy():
        if t in string.punctuation:
            tokens.remove(t)
        elif t in nltk.corpus.stopwords.words("english"):
            tokens.remove(t)
    
    return tokens

    


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    #Dict to keep track of num of doc containing word
    counters = dict()
    for doc in documents:
        #Convert list of words to a set to avoid over counting
        for word in set(documents[doc]):
            #If the word in dict, meaning other doc contains this word
            if word in counters:
                counters[word] += 1
            #Set occurence to 1 if it not in dict
            else:
                counters[word] = 1

    for word in counters:
        #Calculate idf with given function and update value in dict
        idf = math.log(len(documents.keys()) / counters[word])
        counters[word] = idf
            
    return counters
            


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    #List to keeps track of tuple of filename and tfidf sum
    frequencies = []
    for file in files:
        #Initialize dict to track term frequencies 
        tf = dict()
        for word in files[file]:
            #Update dict only when the word is in query
            if word in query:
                #If word already in dict, update it
                if word in tf:
                    tf[word] += 1
                #If word not in dict, initialize it
                else:
                    tf[word] = 1

        sum_tfidf = 0
        #Calculate sum of tf-idf
        for word in tf:
            sum_tfidf += tf[word] * idfs[word]
        #Append a tuple of filename and sum of tf-idf to list
        frequencies.append((file, sum_tfidf))

    #Sort the list based on sum of tf-idf in descending order
    sorted_list = sorted(frequencies, key=lambda freq: freq[1], reverse=True)

    #Return the first n element of the sorted list
    res = []
    for i in range(n):
        res.append(sorted_list[i][0])
    
    return res

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    frequencies = []
    for s in sentences:
        #Initialize sum of idf and count of query term
        sum_idf = 0
        count_qtd = 0 
        for word in query:
            #If the word in the query also in sentence, update sum of idf and count of qtd
            if word in sentences[s]:
                sum_idf += idfs[word]
                count_qtd += sentences[s].count(word)
        #Append tuple of sentence, sum of idf, and query term density to list
        frequencies.append((s, sum_idf, count_qtd / len(sentences[s])))
    #Sort the list based on idf sum with tiebreaker of query term density
    sorted_list = sorted(frequencies, key=lambda freq:(freq[1], freq[2]), reverse = True)

    #Return the first n element of the sorted list
    res = []
    for i in range(n):
        res.append(sorted_list[i][0])

    return res
    


if __name__ == "__main__":
    main()
