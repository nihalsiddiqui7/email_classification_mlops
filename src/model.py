import argparse
import logging
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import BorderlineSMOTE


#making logs directory
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

#making logger
logger = logging.getLogger('model')
logger.setLevel('DEBUG')

# Avoid duplicate handlers when module is imported multiple times
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    log_file_path = os.path.join(log_dir,'model.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel('DEBUG')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)



#loading data function
def load_data(file_path:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df.fillna('',inplace=True)
        logger.debug(f"Data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e

#model training function

def train_model(X_train,y_train,model_type:str,random_state=42):
    try:
        logger.info(f"Training {model_type} model")
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=random_state,n_jobs=-1,n_estimators=100,max_depth=10)
        elif model_type == 'xgboost':
            model = XGBClassifier(random_state=random_state,n_jobs=-1,n_estimators=100,max_depth=10,learning_rate=0.1)
        else:
            raise ValueError("Invalid model type. Supported types are 'random_forest' and 'xgboost'")

        #handling class imbalance using BorderlineSMOTE
        smote = BorderlineSMOTE(random_state=random_state)
        X_train_resampled,y_train_resampled = smote.fit_resample(X_train,y_train)

        model.fit(X_train_resampled,y_train_resampled)
        logger.info(f"{model_type} model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise e

#model saving function
def save_model(model,model_name:str,file_path:str):
    try:
        model_dir = os.path.join(file_path, "models")
        os.makedirs(model_dir,exist_ok=True)
        model_path = os.path.join(model_dir,f"{model_name}.pkl")
        with open(model_path,'wb') as f:
            pickle.dump(model,f)
        logger.info(f"Model saved successfully at {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise e

def main(model_type:str = 'random_forest', file_path:str = '.'):
    try:
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.drop('label',axis=1)
        y_train = train_data['label']
        model = train_model(X_train,y_train,model_type=model_type)
        save_model(model,model_name=model_type,file_path=file_path)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a spam classifier")
    parser.add_argument("--model_type", choices=["random_forest","xgboost"], default="random_forest", help="Type of model to train")
    parser.add_argument("--output_dir", default=".", help="Base directory where the models folder will be created")
    args = parser.parse_args()

    main(model_type=args.model_type, file_path=args.output_dir)