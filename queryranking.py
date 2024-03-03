import pandas as pd
import numpy as np
import re
import sys
import os

import lightgbm as lgb

from IPython.display import display
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report

from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec
from scipy.stats import spearmanr




"""
TODO
- compute clues
- generate dataset of clue values and ranks
- standardize clues (mean and variance)
- train lightgbm model to predict ranks based on clue data
Clues:
QAF: Query Absolute Frequency = Count of term occurrences in the query
QRF: Query Relative Frequency = QAF / total number of occurrences of all terms in the query
DAF: Document Absolute Frequency = Count of term occurrences in the document
DRF: Document Relative Frequency = DAF / total number of occurrences of all terms in the document
IDF: Inverse Document Frequency
RFAD: Relative frequency in all documents = total number of term occurrences of the term in the collection divided by the total number of occurrences of all terms in the entire collection

"""
class QueryRanking: 
    """ 
    Arguments: 
    queries - list of query inputs that are responsible for a specific data ranking. 
    Must be an excel sheet in the same file where the name is the query. The format has to match the sheets provided.
    """
    def __init__(self) -> None:
        self.model = None
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.train_queries = ['glucose in blood', 'bilirubin in plasma', 'White blood cells count', 'Serum cholesterol', 'Liver enzyme']
        self.column_names = ['loinc_num', 'loinc_common_name', 'component', 'system', 'property']

        # initialize tf-idf vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

        # Load pre-trained Word2Vec model
        #self.word2vec_model = Word2Vec.load('GoogleNews-vectors-negative300.bin')

        # load excel file
        dataset = pd.read_excel("extended_dataset_.xlsx", sheet_name=self.train_queries, header=2, names=self.column_names)

        # save data into dictionary where the name is the query and the value is a dataframe containing the ranking results.
        self.data = {}
        for query in self.train_queries:
            self.data[query] = dataset[query].copy()

            # create new attribute content that contains the content from all columns as one string
            self.data[query]['content'] = self.data[query].apply(lambda row: ' '.join(row), axis=1)
            self.data[query]['content'] = self.data[query]['content'].apply(self.clean_string)
            self.data[query]['content'] = self.data[query]['content'].str.lower()
            
            # create new attribute rank that describes the rank that each document has for the specific query
            self.data[query]['rank'] = len(self.data[query].index) - self.data[query].index
            # print(query)
            # print(self.data[query].head(5))

    
    def train_model(self):
        """
        Train model based on data provided
        Saves the trained model in the variable self.model
        """
        model_data = pd.DataFrame(columns=['tfidf_score', 'bm25_score', 'rank'])
        groups = []
        for query in self.train_queries:
            data = self.data[query]

            # generate clue data
            clue_data = self.generate_clue_data(data['content'], query)

            # copy ranks from original data
            clue_data['rank'] = data['rank']

            # assign query group
            groups.append(len(clue_data))

            # append query specific data to overall dataframe 
            model_data = pd.concat([model_data, clue_data], ignore_index=True)

        # Standardize values
        scaler = StandardScaler()
        model_data.iloc[:, :-2] = scaler.fit_transform(model_data.iloc[:, :-2])

        # Labelencode rank
        encoder = LabelEncoder()
        model_data['rank'] = encoder.fit_transform(model_data['rank'].astype('category'))

        # Split data
        X = model_data[['tfidf_score', 'bm25_score']]
        y = model_data['rank']  # Target

        # Initialize and train model
        self.model = lgb.LGBMRanker(
            label_gain=[i for i in range(y.max() + 1)]
        ) 
        
        self.model.fit(X, y, group=groups)  # Train the model

        self.get_train_error()


        
    def predict_ranking(self, rank_data: pd.Series, query: str) -> pd.DataFrame:
        """
        Predict by ranking an input Series of documents, converting it to clue data and predicting using the model
        Input: 
            rank_data: Series of documents
            query: String containing the query
        Output: 
            Dataframe that contains the predicted ranks for the input documents and the query.
        """
        if self.model is None:
            self.train_model()  # Train the model if not already trained
        
        result = pd.DataFrame()
        result['content'] = rank_data.copy()
        # Generate clue data for testing data to apply the model to
        data = self.generate_clue_data(rank_data, query)

        scaler = StandardScaler()

        # Standardize input data
        data = scaler.fit_transform(data)

        # Predict ranks with the trained model
        predictions = self.model.predict(data)

        # Add predictions to result dataframe
        result['predicted_rank'] = predictions

        # transform predictions into integer ranks
        result['predicted_rank'] = result['predicted_rank'].rank(method='dense', ascending=False).astype(int)

        return result
    
    
    def generate_clue_data(self, rank_data: pd.Series, query: str) -> pd.DataFrame:
        """
        Generate clue data for an input series of documents.
        Input:
            rank_data - Series that contains documents
            query - String that contains the query
        Output:
            Dataframe that contains the score metrics for each document-query pair
        """
        # initialize dataframe
        data_columns = ['tfidf_score', 'bm25_score']
        data = pd.DataFrame(columns=data_columns)

        # clean query from special characters
        clean_query = self.clean_string(query)

        # generate column data
        data['tfidf_score'] = self.tfidf_similarity_score(rank_data, clean_query)
        data['bm25_score'] = self.bm25_similarity_score(rank_data, clean_query)
        #data['embedding_score'] = self.embedding_score(rank_data, clean_query)
        return data
    
    
    def evaluate_performance(self, ranking: pd.Series, prediction: pd.Series):
        """
        TODO
        Some function to calculate precision, recall etc on unseen data that labels are known for
        """
        pass

    def get_train_error(self):
        """
        TODO
        Calculate performance on seen data
        """
        # load subset of training data
        query = self.train_queries[0]
        train_results = self.predict_ranking(self.data[query]['content'], query)
        train_results['correct_rank'] = self.data[query]['rank']

        print("Spearman Correlation:")
        print(spearmanr(train_results['predicted_rank'], train_results['correct_rank']).correlation)
        
    

    def embedding_score(self, rank_data: pd.Series, query: str) -> pd.Series:
        """
        Input: 
            Series of documents
            str query

        Output:
            Series of cosine similarity scores between the query and each document using word embeddings.
        """
        # Tokenize the query and documents
        tokenized_query = query.split()
        tokenized_documents = [doc.split() for doc in rank_data]

        # Get Word2Vec model
        word2vec_model = self.word2vec_model

        # Generate embeddings for the query and documents
        query_embedding = [word2vec_model[word] for word in tokenized_query if word in word2vec_model.wv.vocab]
        document_embeddings = [[word2vec_model[word] for word in doc if word in word2vec_model.wv.vocab] for doc in tokenized_documents]

        # Aggregate token embeddings for documents
        aggregated_document_embeddings = [sum(doc_embedding) / len(doc_embedding) for doc_embedding in document_embeddings]

        # Calculate similarity scores using cosine similarity
        similarity_scores = [cosine_similarity([query_embedding], [doc_embedding])[0][0] for doc_embedding in aggregated_document_embeddings]
        return similarity_scores
    
    
    def tfidf_similarity_score(self, rank_data: pd.Series, query: str) -> pd.Series:
        """
        Input: 
            Series of documents
            str query

        Output:
            Series of cosine similarity scores between the query and each document
        """
        score_list = []
        for idx, document in rank_data.items():
            # create tfidf matrix
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([query, document])

            # Get TF-IDF vectors for query and document
            query_tfidf_vector = tfidf_matrix[0].toarray().flatten()  # TF-IDF vector for query
            document_tfidf_vector = tfidf_matrix[1].toarray().flatten()  # TF-IDF vector for document

            score_list.append(cosine_similarity([query_tfidf_vector], [document_tfidf_vector])[0][0])

        return pd.Series(score_list)
    
    
    def bm25_similarity_score(self, rank_data: pd.Series, query: str) -> pd.Series:
        """
        Input: 
            Series of documents
            str query

        Output:
            Series of bm25 similarity scores between the query and each document
        """
        # Tokenize the query and documents
        tokenized_query = query.split()
        tokenized_documents = [doc.split() for doc in rank_data]

        # Create a corpus of documents
        corpus = tokenized_documents

        # Initialize BM25 object with the corpus
        bm25 = BM25Okapi(corpus)

        # Calculate BM25 scores for each document
        scores = bm25.get_scores(tokenized_query)

        return pd.Series(scores)

    
    def clean_string(self, input_string: str) -> str:
        """
        Helper function to clean the string from double spaced, leading and trailing spaces and special characters
        """
        # Remove special characters except '#'
        cleaned_string = re.sub(r'[^\w\s#]', ' ', input_string)
        # Collapse consecutive spaces into a single space
        cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
        # Strip leading and trailing spaces
        cleaned_string = cleaned_string.strip()
        cleaned_string = cleaned_string.lower()
        return cleaned_string







def main():
    """
    Called when executing the file. 
    When executed through the terminal, no parameters are required.
    """

    qr = QueryRanking()

    while True:
        print("An Excel file is needed that contains a sheet 'query' that contains a list of items in the correct format")
        query_path = input("Add the path for the data to be ranked: ")
        while True:
            if not os.path.isfile(query_path) or not query_path.endswith(".xlsx"):
                query_path = input("File does not exist or has the wrong format, please enter valid file path (.xlsx): ").strip()
                continue
            else:
                try:
                    # load query data
                    query_data = pd.read_excel(query_path, sheet_name=['query'], header=2, names=qr.column_names)
                    break
                except:
                    print("Invalid excel document. Please check the requirements and adjust the file.")
                    continue

        query = input("Enter the query: ").strip()

        
        data = query_data['query'].copy()

        # create new attribute content that contains the content from all columns as one string
        data['content'] = data.apply(lambda row: ' '.join(row), axis=1)
        data['content'] = data['content'].apply(qr.clean_string)
        
        # create new attribute rank that describes the rank that each document has for the specific query
        data['rank'] = len(data.index) - data.index

        # clean query string
        query = qr.clean_string(query)

        # predict ranks
        result = qr.predict_ranking(data['content'], query)

        result['correct_rank'] = data['rank']
        result = result.sort_values(by='predicted_rank', ascending=False)
        print(result.head(10))

if __name__ == "__main__":
    main()