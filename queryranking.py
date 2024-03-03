import pandas as pd
import numpy as np
import re

from lightgbm import LGBMRanker
from IPython.display import display
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec




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
    def __init__(self, queries) -> None:
        self.model = None
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.queries = queries
        self.column_names = ['loinc_num', 'loinc_common_name', 'component', 'system', 'property']

        # initialize tf-idf vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

        # Load pre-trained Word2Vec model
        #self.word2vec_model = Word2Vec.load('GoogleNews-vectors-negative300.bin')

        # load excel file
        dataset = pd.read_excel("loinc_dataset-v2.xlsx", sheet_name=queries, header=2, names=self.column_names)

        # save data into dictionary where the name is the query and the value is a dataframe containing the ranking results.
        self.data = {}
        for query in self.queries:
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
        TODO
        Train model based on data provided
        """
        model_data = pd.DataFrame(columns=['tfidf_score', 'bm25_score', 'rank'])
        groups = []
        for query in self.queries:
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
        self.model = LGBMRanker(
            label_gain=[i for i in range(y.max() + 1)]


        ) # Initialize Light GBM Ranker model
        self.model.fit(X, y, group=groups)  # Train the model

        
    def predict_ranking(self, rank_data: pd.Series, query: str) -> pd.DataFrame:
        """
        TODO
        Predict by ranking an input dataframe, converting it to clue data and predicting using the model
        """
        if self.model is None:
            self.train_model()  # Train the model if not already trained
        
        result = rank_data.copy()
        # Generate clue data for testing data to apply the model to
        data = self.generate_clue_data(rank_data, query)

        scaler = StandardScaler()

        # Standardize input data
        data = scaler.fit_transform(data)

        # Predict ranks with the trained model
        predictions = self.model.predict(data)

        # Add predictions to result dataframe
        result['predicted_rank'] = predictions

        return result
    
    """
    Generate clue data for an input dataframe
    """
    def generate_clue_data(self, rank_data: pd.Series, query: str) -> pd.DataFrame:
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
    
    """
    TODO
    Some function to calculate precision, recall etc on unseen data that labels are known for
    """
    def evaluate_performance(self, ranking: pd.Series, prediction: pd.Series):
        pass


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

    
    """
    Adjust clue value column by standardizing with mean and standard deviation
    """
    def standardize_values(self, column: pd.Series) -> pd.Series:
        return (column - column.mean()) / column.std()
    
    
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
        return cleaned_string







"""
Main function to create QueryRanking object and generate model
"""
def main():
    queries = ['glucose in blood', 'bilirubin in plasma', 'White blood cells count']
    qr = QueryRanking(queries)

    # load query data
    query_data = pd.read_excel("Serum_cholesterol_.xlsx", sheet_name=['query'], header=2, names=qr.column_names)

    print(query_data)
    data = query_data['query'].copy()

    # create new attribute content that contains the content from all columns as one string
    data['content'] = data.apply(lambda row: ' '.join(row), axis=1)
    data['content'] = data['content'].apply(qr.clean_string)

    
    # create new attribute rank that describes the rank that each document has for the specific query
    data['rank'] = len(data.index) - data.index
    # print(query)
    # print(self.data[query].head(5))
    result = qr.predict_ranking(data['content'], 'Serum cholesterol')

    result['correct_rank'] = data['rank']

    print(result.head())
    """
    # change when needed
    column_names = ['loinc_num', 'loinc_common_name', 'component', 'system', 'property']
    data = pd.DataFrame(columns=column_names)
    query = "example"
    # get some list of data from somewhere

    results = qr.predict_ranking(data, query)
    display(results)

    # evaluate accuracy using the function evaluate_performance()
    """


if __name__ == "__main__":
    main()