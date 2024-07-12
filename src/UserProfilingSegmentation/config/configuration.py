

from src.UserProfilingSegmentation.constants import *
from src.UserProfilingSegmentation.utils.common import read_yaml, create_directories
from src.UserProfilingSegmentation.entity.config_entity import (DataIngestionConfig, DataValidationConfig)
from src.UserProfilingSegmentation import logger




class ConfigurationManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        logger.info(f'----creating the artifacts_root---')

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        logger.info(f'-------Entered into get_data_ingestion_config method-----------------')
        config = self.config.data_ingestion

        create_directories([config.root_dir])


        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            local_data_file = config.local_data_file
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        
        config = self.config.data_validation
        schema = self.schema.COLUMNS


        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir = config.root_dir,
            local_data_file= config.local_data_file,
            STATUS_FILE = config.STATUS_FILE,
            all_schema = schema
        )

        return data_validation_config