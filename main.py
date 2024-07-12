
import sys
from src.UserProfilingSegmentation import logger, CustomException
from src.UserProfilingSegmentation.pipeline.stage_01_dataIngestion import DataIngestionTrainingPipeline



STAGE_NAME = 'Data Ingestion Stage'


try:
    logger.info(f'--------------------stage {STAGE_NAME} started --------------------------')
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f'----------------------stage {STAGE_NAME} completed ---------------------')
except Exception as e:
    raise CustomException(e, sys)