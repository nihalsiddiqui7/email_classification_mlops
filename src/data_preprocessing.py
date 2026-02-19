import pandas as pd
import os
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re


#making logs directory
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

#making logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

#defining handler 1-Console handler which will print logs to console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#defining handler 2-File handler which will save logs to a file
log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#defining formatter and adding it to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


#data preprocessing function
def clean_text(text):
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        # Stemming
        porter_stemmer = PorterStemmer()
        text = ' '.join([porter_stemmer.stem(word) for word in text.split()])
        return text
    except Exception as e:
        logger.error(f"Error cleaning text data: {e}")
        raise e

# function to clean data as there as duplicates in the data

def preprocess_data(df,text_col = 'message',label_col = 'label'):
    try:
        #dropping duplicates
        df.drop_duplicates(inplace=True)
        df[text_col] = df[text_col].apply(clean_text)
        return df
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise e


#main function to preprocess data and save it as preprocessed train and test

def main(text_col = 'message',label_col = 'label'):
    try:
        train_df = pd.read_csv("./data/processed/train.csv")
        test_df = pd.read_csv("./data/processed/test.csv")
        logger.info("Preprocessing train data")
        train_df_processed = preprocess_data(train_df,text_col=text_col,label_col=label_col)
        logger.info("Preprocessing test data")
        test_df_processed = preprocess_data(test_df,text_col=text_col,label_col=label_col)
        logger.info("Starting data preprocessing")

        train_df_processed = preprocess_data(train_df,text_col='message',label_col='label')
        test_df_processed = preprocess_data(test_df,text_col='message',label_col='label')
        logger.info("Data preprocessing completed successfully")

        data_path = os.path.join("data", "processed")
        train_df_processed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_df_processed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.info("Preprocessed data saved successfully to:", os.path.join(data_path, "train_processed.csv"), "and", os.path.join(data_path, "test_processed.csv"))
    except Exception as e:
        logger.error(f"Error in data preprocessing process: {e}")
        raise e
if __name__ == "__main__":
    main()

