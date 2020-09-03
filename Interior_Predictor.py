import sys

import nltk
import pandas as pd
import numpy as np
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from array import *


def get_comments():
    column = df.loc[:, 'Public Remarks']
    commentList = [None] * len(column)
    for i in range(len(column)):
        commentList[i] = column[i]
    return commentList


def get_interior_ratings():
    column = df.loc[:, 'Interior']
    ratings = np.zeros(len(column))
    for i in range(len(column)):
        ratings[i] = column[i]
    return ratings


def get_listing_vals():
    column = df.loc[:, 'Listing_eval']
    evaluations = [None] * len(column)
    for i in range(len(column)):
        evaluations[i] = column[i]
    return evaluations
        
        
def get_word_tokens(com):
    allTokens = []
    for i in range(len(com)):
        tokens = word_tokenize(str(com[i]))
        allTokens.append(tokens)
    return allTokens


def get_sentences(com):
    sentences = []
    for i in com:
        sentences.append(i)
    return sentences


def remove_stop_words_and_punctuation(tokens):
    stopWords = set(stopwords.words("english"))
    tokenSet = []

    for i in tokens:
        iterations = 0
        arrTokens = []
        for j in i:
            if j.lower() not in stopWords and j.isalpha():
                arrTokens.append(j)
        tokenSet.append(arrTokens)

    return tokenSet


def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    list_lematized = []
    for i in range(len(tokens)):
        lemmatized_token = []
        for j in range(len(tokens[i])):
            lemmatized_token.append(lemmatizer.lemmatize(tokens[i][j]))
        list_lematized.append(lemmatized_token)
    return list_lematized


def stemmer(tokens):
    ps = PorterStemmer()
    list_stemmed = []
    for i in range(len(tokens)):
        stemmed_token = []
        for j in range(len(tokens[i])):
            stemmed_token.append(ps.stem(tokens[i][j]))
        list_stemmed.append(stemmed_token)
    return list_stemmed


def get_all_words(filtered_tokens):  # Getting all words to create our doc-term matrix
    words = []
    for i in range(len(filtered_tokens)):
        for j in range(len(filtered_tokens[i])):
            if filtered_tokens[i][j].lower() in words:
                continue
            else:
                words.append(filtered_tokens[i][j].lower())

    return words


def doc_word_counts(filter_tokens, words):
    matrix = []
    for i in range(len(filter_tokens)):
        counts = [0] * (len(words))

        for j in range(len(filter_tokens[i])):
            counts[words.index(filter_tokens[i][j].lower())] += 1

        matrix.append(counts)

    return matrix


def tf_idf(matrix, words):
    docSize = []
    for i in range(len(matrix)):  # document sizes
        docCount = 0
        working = 0
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                docCount = docCount + matrix[i][j]
        docSize.append(docCount)

    docAppearanceCount = [0] * len(words)
    for i in range(len(words)):  # number of documents each word appears in
        for j in range(len(matrix)):
            if matrix[j][i] != 0:
                docAppearanceCount[i] += 1
                continue
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            tf = matrix[i][j] / docSize[i]
            idf = math.log(len(matrix) / docAppearanceCount[j])
            matrix[i][j] = tf * idf
    return matrix


def initialize_clusters(doc_matrix):
    initial_cluster = [doc_matrix[3], doc_matrix[1], doc_matrix[0]]

    return initial_cluster


def cosine_similarity(arr1, arr2):
    topFrac = 0
    bottom1 = 0;
    bottom2 = 0;

    for i in range(len(arr1)):
        topFrac += arr1[i] * arr2[i]
        bottom1 += arr1[i] ** 2
        bottom2 += arr2[i] ** 2
    similarity = topFrac / (math.sqrt(bottom1) * math.sqrt(bottom2))
    return similarity


def update(current_clusters, current_indexes, doc_matrix, number_of_clusters):
    clusterSizes = []
    current_clusters.clear()
    for i in range(number_of_clusters):
        a = np.array(current_indexes)
        clusterSizes.append(np.count_nonzero(a == i))
        values = [0] * len(doc_matrix[0])
        for j in range(len(doc_matrix)):

            for k in range(len(doc_matrix[j])):
                if current_indexes[j] == i:
                    values[k] += doc_matrix[j][k]
        current_clusters.append(values)

    for i in range(len(current_clusters)):
        for j in range(len(doc_matrix[0])):
            current_clusters[i][j] /= clusterSizes[i]


def k_means(matrix):
    cluster = initialize_clusters(matrix)
    indexes = [0] * len(matrix)
    changed = True
    while changed:
        changed = False
        for i in range(len(matrix)):
            maximum = -99999999999
            docClusterIndex = indexes[i]
            maxIndex = 0
            for j in range(len(cluster)):

                cosSimilarity = cosine_similarity(cluster[j], matrix[i])
                if cosSimilarity > maximum:
                    maximum = cosSimilarity
                    maxIndex = j
            if docClusterIndex != maxIndex:
                indexes[i] = maxIndex
                changed = True
        update(cluster, indexes, matrix, len(cluster))
    return cluster, indexes


def print_indexes(docs):
    for i in range(len(docs)):
        print(docs[i])


def average_interior(indexes, ratings, cluster):
    for i in range(len(cluster)):
        interior = 0
        numCluster = 0
        for j in range(len(indexes)):
            if indexes[j] == i:
                interior += ratings[j]
                numCluster += 1

        print(interior/numCluster)


def counts_of_listing_eval(cluster, listings, indexes):
    for i in range(len(cluster)):
        under_count = 0
        over_count = 0
        correct_count = 0
        cluster_count = 0
        for j in range(len(listings)):
            if indexes[j] == i:
                cluster_count += 1
                if listings[j] == 'under':
                    under_count += 1
                elif listings[j] == 'over':
                    over_count += 1
                else:
                    correct_count += 1
        print("Under: " + str(under_count), "Correct: " + str(correct_count), "Over: " + str(over_count))
        print(under_count/cluster_count, correct_count/cluster_count, over_count / cluster_count)


def representative_words(cluster, all_words):
    represents = []
    for i in range(len(cluster)):
        a = np.array(cluster[i])
        idx = (-a).argsort()[:3]
        words = [all_words[idx[0]], all_words[idx[1]], all_words[idx[2]]]
        represents.append(words)

    print(represents)


with open('sentimentAnalysis.csv', 'rb') as csv_file:
    df = pd.read_csv(csv_file, delimiter=',',
                     usecols=["Interior", "Public Remarks", "Listing_eval"], encoding='ASCII')
    interiorRatings = get_interior_ratings()

    comments = get_comments()
    listing_val = get_listing_vals()
    listing = np.array(listing_val)
    print(np.where(listing == 'over'))
    print(np.where(listing == 'under'))
    print(np.where(listing == 'correct'))
    # pre-processing
    tokenizedWords = get_word_tokens(comments)  # List of all tokens
    tokenizedSentences = get_sentences(comments)
    filteredSet = remove_stop_words_and_punctuation(tokenizedWords)
    # lemmatize_set = lemmatize(filteredSet)
    # stemmed_set = stemmer(lemmatize_set)
    allWords = get_all_words(filteredSet)
    docTermMatrix = doc_word_counts(filteredSet, allWords)

    tfIdfMatrix = tf_idf(docTermMatrix, allWords)
    clusters, docIndexes = k_means(tfIdfMatrix)

    average_interior(docIndexes, interiorRatings, clusters)
    counts_of_listing_eval(clusters, listing_val, docIndexes)
    representative_words(clusters, allWords)
