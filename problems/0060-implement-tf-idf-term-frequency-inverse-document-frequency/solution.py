import torch
import math
from collections import Counter
from typing import List

def compute_tf_idf(corpus: List[List[str]], query: List[str]) -> torch.Tensor:
    """
    Compute TF-IDF scores for a query against a corpus of documents using PyTorch.

    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: torch.Tensor of shape (num_docs, num_query_words) with TF-IDF scores
             rounded to five decimal places
    """
    N = len(corpus)

    # Count term frequencies in each document
    doc_counts = [Counter(doc) for doc in corpus]
    print(doc_counts)

    # Document frequency: number of documents containing each word
    df = Counter()

    for counts in doc_counts:
        for word in counts:
            df[word] += 1

    # Compute TF-IDF scores
    scores = []

    for j, doc in enumerate(corpus):

        doc_len = len(doc)
        doc_scores = []

        for word in query:

            # Term frequency
            tf = doc_counts[j][word] / doc_len if doc_len > 0 else 0.0

            # Smoothed IDF
            idf = math.log((N + 1) / (df[word] + 1)) + 1

            # TF-IDF
            score = tf * idf

            doc_scores.append(round(score, 5))

        scores.append(doc_scores)

    return torch.tensor(scores, dtype=torch.float32)
    
    
