The data consists of 2225 documents from the BBC news website corresponding to stories in five topical 
areas from 2004-2005. Natural Classes: 5 (business, entertainment, politics, sport, tech)

Files in this archive:
---------------------

"bbc-5categories.csv": This file contains the full raw data set. It contains columns for document id, 
document title, document category (class label), category id, and the full text.

"BBC_News_5_TF.csv": This file contains the 2225 x 6167 document x term matrix of term frequency values. 
Each row corresponds to a document and each column corresponds to a feature (features are a subset of 
the stemmed and normalized term tokens extracted from the full text). Note that the first column in this 
data represents the document ids and the first row represents the feature id; these are not part of the 
document x term matrix used for modeling.

"BBC_News_5_Classes.csv": This file contains the categories (class labels) for each document. The first 
column represents the document id and the second column contains the class label for that document.

"BBC_News_5_Features.csv": This file contains the list of all 6167 features used to represent documents 
in the document x term matrix. Each feature is listed in one row. There is no index column, so the first 
row contains feature 0, the second row contains feature 1, and so on. 

