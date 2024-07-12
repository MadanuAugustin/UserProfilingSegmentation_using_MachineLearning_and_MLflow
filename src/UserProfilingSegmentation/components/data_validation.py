import sys
import pandas as pd
from src.UserProfilingSegmentation.entity.config_entity import DataValidationConfig
from src.UserProfilingSegmentation import logger, CustomException


class DataValidation:
    def __init__(self, config : DataValidationConfig):
        self.config = config



    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.local_data_file)

            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"validation status : {validation_status}")
                
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f'validation status : {validation_status}')

            return validation_status
        
        except Exception as e:
            raise CustomException(e, sys)