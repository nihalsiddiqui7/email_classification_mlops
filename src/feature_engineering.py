from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import logging

#making logs directory
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

#making logger
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

#defining handler 1-Console handler which will print logs to console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#defining handler 2-File handler which will save logs to a file
log_file_path = os.path.join(log_dir,'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#defining formatter and adding it to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


#feature engineering function

def load_data(file_path:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df.fillna('',inplace=True)
        logger.debug(f"Data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e

def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int):
    try:
        logger.info("Applying TF-IDF vectorization")
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)


        X_train = train_data['message'].values
        y_train = train_data['label'].values
        X_test = test_data['message'].values
        y_test = test_data['label'].values

        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        train_data_tfidf = pd.DataFrame(X_train_tfidf.toarray(),columns=tfidf_vectorizer.get_feature_names_out())
        train_data_tfidf['label'] = y_train
        test_data_tfidf = pd.DataFrame(X_test_tfidf.toarray(),columns=tfidf_vectorizer.get_feature_names_out())
        test_data_tfidf['label'] = y_test

        logger.info("TF-IDF vectorization applied successfully")
        return train_data_tfidf,test_data_tfidf
    except Exception as e:
        logger.error(f"Error applying TF-IDF vectorization: {e}")
        raise e

def save_data(df:pd.DataFrame,file_path:str):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.info(f"Data saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise e


def main():
    try:
        max_features = 7000
        train_data = load_data("./data/processed/train_processed.csv")
        test_data = load_data("./data/processed/test_processed.csv")
        train_data_tfidf,test_data_tfidf = apply_tfidf(train_data,test_data,max_features)
        save_data(train_data_tfidf,os.path.join("data", "processed", "train_tfidf.csv"))
        save_data(test_data_tfidf,os.path.join("data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.error(f"Error in feature engineering process: {e}")
        raise e
if __name__ == "__main__":
    main()