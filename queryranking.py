import pandas as pd
import re

from lightgbm import LGBMRanker
from IPython.display import display



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

        # load excel file
        dataset = pd.read_excel("loinc_dataset-v2.xlsx", sheet_name=queries, header=2, names=self.column_names)

        # save data into dictionary where the name is the query and the value is a dataframe containing the ranking results.
        self.data = {}
        for query in self.queries:
            self.data[query] = dataset[query].copy()

            # create new attribute content that contains the content from all columns as one string
            self.data[query]['content'] = self.data[query].apply(lambda row: ' '.join(row), axis=1)
            self.data[query]['content'] = self.data[query]['content'].apply(self.clean_string)

            
            # create new attribute rank that describes the rank that each document has for the specific query
            self.data[query]['rank'] = len(self.data[query].index) - self.data[query].index

    """
    TODO
    Train model based on data provided
    """
    def train_model(self):
        model_data = pd.DataFrame(columns=['qaf', 'qrf', 'daf', 'drf', 'idf', 'rfad', 'rank'])
        for query in self.queries:
            data = self.data[query]

            # generate clue data
            clue_data = self.generate_clue_data(data['content'], query)
            # copy ranks from original data
            clue_data['rank'] = data['rank']

            # append query specific data to overall dataframe 
            model_data = pd.concat([model_data, clue_data], ignore_index=True)

        # normalize values
        # initialize and train model
        # save model to self.model

    
    """
    TODO
    Predict by ranking an input dataframe, converting it to clue data and predicting using the model
    """
    def predict_ranking(self, rank_data: pd.Series, query: str) -> pd.DataFrame:
        if self.model == None:
            self.train_model()
        
        result = rank_data.copy()
        # generate clue data for testing data to apply the model to
        data = self.generate_clue_data(rank_data, query)

        # predict ranks with model
        # add predictions to result dataframe
        return result
        
    
    """
    Generate clue data for an input dataframe
    """
    def generate_clue_data(self, rank_data: pd.Series, query: str) -> pd.DataFrame:
        # initialize dataframe
        data_columns = ['qaf', 'qrf', 'daf', 'drf', 'idf', 'rfad']
        data = pd.DataFrame(columns=data_columns)

        # clean query from special characters
        clean_query = self.clean_string(query)

        # generate column data
        data['qaf'] = self.query_absolute_frequency(rank_data, clean_query)
        data['qrf'] = self.query_relative_frequency(rank_data, clean_query)
        data['daf'] = self.document_absolute_frequency(rank_data, clean_query)
        data['drf'] = self.document_relative_frequency(rank_data, clean_query)
        data['idf'] = self.inverse_document_frequency(rank_data, clean_query)
        data['rfad'] = self.relative_frequency_all_documents(rank_data, clean_query)
        return data
    
    """
    TODO
    Some function to calculate precision, recall etc on unseen data that labels are known for
    """
    def evaluate_performance(self):
        pass

    """
    TODO
    """
    def query_absolute_frequency(self, rank_data: pd.Series, query: str) -> pd.Series:
        pass

    """
    TODO
    """
    def query_relative_frequency(self, rank_data: pd.Series, query: str) -> pd.Series:
        pass

    """
    TODO
    """
    def document_absolute_frequency(self, rank_data: pd.Series, query: str) -> pd.Series:
        pass

    """
    TODO
    """
    def document_relative_frequency(self, rank_data: pd.Series, query: str) -> pd.Series:
        pass
    
    """
    TODO
    """
    def inverse_document_frequency(self, rank_data: pd.Series, query: str) -> pd.Series:
        pass
    
    """
    TODO
    """
    def relative_frequency_all_documents(self, rank_data: pd.Series, query: str) -> pd.Series:
        pass
    
    """
    TODO
    Adjust clue value column by standardizing with mean and standard deviation
    """
    def standardize_values(self, column: pd.Series) -> pd.Series:
        pass
    
    """
    Helper function to clean the string from double spaced, leading and trailing spaces and special characters
    """
    def clean_string(self, input_string: str) -> str:
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