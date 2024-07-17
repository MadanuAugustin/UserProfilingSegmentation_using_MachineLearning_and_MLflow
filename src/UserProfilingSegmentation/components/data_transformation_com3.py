
import os
import sys
import numpy as np
import pandas as pd
import joblib
from src.UserProfilingSegmentation import logger, CustomException
from src.UserProfilingSegmentation.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer





class DataTransformation:
    def __init__(self, config : DataTransformationConfig):
        
        self.config = config




    def feature_selection(self):
        try:

            logger.info(f'-----------Entered data_split method---------------')
        
            data = pd.read_csv(self.config.local_data_file)

            data = data[['Age', 'Gender', 'IncomeLevel', 'TimeSpentOnlinehrsweekday', 'TimeSpentOnlinehrsweekend', 'LikesandReactions', 'ClickThroughRates']]

            data.dropna(inplace=True)
        
            data.to_csv(os.path.join(self.config.data_path, 'data.csv'), index = False, header = True)

            logger.info(f'----------------saved data in csv format------------------')

            logger.info(f'------------The shape of the data is {data.shape}')

        except Exception as e:
            raise CustomException(e, sys)
        

    

    def preprocessor_fun(self):
        try:

            logger.info(f'---------------Entered preprocessor function------------------')


            numeric_columns = ['TimeSpentOnlinehrsweekday', 'TimeSpentOnlinehrsweekend', 'LikesandReactions', 'ClickThroughRates']
            
            categoric_columns = ['Age', 'Gender', 'IncomeLevel']

            logger.info(f'----------creating transformer pipelines---------------')

            numeric_pipeline = Pipeline(
                steps=[
                    ('standardscaler', StandardScaler())
                ]
            )

            categoric_pipeline = Pipeline(
                steps=[
                    ('onhotcoder', OneHotEncoder(drop='first'))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('numericpipeline', numeric_pipeline, numeric_columns),
                    ('categoricpipeline', categoric_pipeline, categoric_columns)
                ]
            )

            logger.info(f'---------------completed creating transformer pipelines---------------')

            logger.info(f'--------------completed preprocessor function------------------')

            return preprocessor
            

        except Exception as e:
            raise CustomException(e, sys)
        

    
    def initiate_data_transformation(self):

        try:

            logger.info(f'------------started initiate_data_transformation method------------')

            data_df = pd.read_csv('artifacts//data_transformation//data.csv')
            
            logger.info(f'----------obtaining the preprocessor obj-----------')

            preprocessor_obj = self.preprocessor_fun()

            transformed_data_df = pd.DataFrame(preprocessor_obj.fit_transform(data_df))

            joblib.dump(preprocessor_obj, os.path.join(self.config.root_dir, 'preprocessor_obj.joblib'))

            # transformed_data_df.rename(columns={0 : 'TimeSpentOnlinehrsweekday', 1 : 'TimeSpentOnlinehrsweekend', 2 : 'LikesandReactions',
            #                                     3 : 'ClickThroughRates', 4 : 'Age', 5 : 'Gender', 6 : 'IncomeLevel'}, inplace=True)

            transformed_data_df.to_csv(os.path.join(self.config.root_dir, 'transformed_data_df.csv'), index = False, header = True)

            logger.info(f'-------------transformed data using preprocessor obj and saved in csv format----------')

            logger.info(f'--------------completed initiate_data_transformation method--------------')

        except Exception as e:
            raise CustomException(e, sys)