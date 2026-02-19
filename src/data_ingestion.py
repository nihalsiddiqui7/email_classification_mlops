import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split


#making logs directory
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)



#making logger

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

#defining handler 1-Console handler which will print logs to console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#defining handler 2-File handler which will save logs to a file
log_file_path = os.path.join(log_dir,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#defining formatter and adding it to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


#data ingestion function
def load_data(file_path:str) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from{file_path}")
        df = pd.read_csv(file_path,encoding='latin-1')
        logger.info(f"Data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e


#splitting data into train and test

def split_data(df,test_size=0.2,random_state=42):
    try:
        logger.info("Splitting data into train and test sets")
        X = df['message']
        y = df['label']
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
        logger.info(f"Data split successfully with train shape {X_train.shape} and test shape {X_test.shape}")
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise e


#i will try to save the data too in this file for now, but later i will move it to a separate file

#preprocessing function

def preprocess_df(df:pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Preprocessing data")
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        logger.info("Data preprocessed successfully.Removed unnecessary columns and mapped labels")
        return df
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise e


#saving data function
def save_data(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame):
    try:
        #create directories if not exist
        raw_data_path = os.path.join("data", "raw")
        processed_data_path = os.path.join("data", "processed")
        os.makedirs(raw_data_path, exist_ok=True)
        os.makedirs(processed_data_path, exist_ok=True)

        # Save raw df
        raw_file_path = os.path.join(raw_data_path, "raw.csv")
        logger.info(f"Saving raw data to {raw_file_path}")
        df.to_csv(raw_file_path, index=False)

        # Save train and test
        train_file_path = os.path.join(processed_data_path, "train.csv")
        test_file_path = os.path.join(processed_data_path, "test.csv")
        logger.info(f"Saving train data to {train_file_path}")
        train_df.to_csv(train_file_path, index=False)
        logger.info(f"Saving test data to {test_file_path}")
        test_df.to_csv(test_file_path, index=False)

        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise e

#main function to run the whole data ingestion process

def main():
    try:
        file_path = "D:\\Data science\\Portfolio Projects\\email_spam\\email_classification_mlops\\spam.csv"           # root folder, not data/spam.csv
        raw_df = load_data(file_path)
        df = preprocess_df(raw_df)       # keep raw_df separate
        X_train, X_test, y_train, y_test = split_data(df)
        train_df = pd.DataFrame({'message': X_train, 'label': y_train})
        test_df = pd.DataFrame({'message': X_test, 'label': y_test})
        save_data(raw_df, train_df, test_df)  # pass raw_df, not df
    except Exception as e:
        logger.error(f"Error in data ingestion process: {e}")
        raise e

if __name__ == "__main__":
    main()