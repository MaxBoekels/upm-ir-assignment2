import pandas as pd

queries = ['glucose in blood', 'bilirubin in plasma', 'White blood cells count']
column_names = ['loinc_num', 'loinc_common_name', 'component', 'system', 'property']

# load excel file
dataset = pd.read_excel("loinc_dataset-v2.xlsx", sheet_name=queries, header=2, names=column_names)

query1 = dataset['glucose in blood']
query2 = dataset['bilirubin in plasma']
query3 = dataset['White blood cells count']


query1['rank1'] = len(query1.index) - query1.index
query2['rank2'] = len(query2.index) - query2.index
query3['rank3'] = len(query3.index) - query3.index

print(query1)
print("----------")
print(query2)
print("----------")
print(query3)
print("----------")

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