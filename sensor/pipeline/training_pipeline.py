from sensor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from sensor.exception import SensorException
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.components.data_ingestion import DataIngestion
import sys
from sensor.logger import logging


class TrainPipeline:

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        
    
    def start_data_ingestion(self)->DataIngestionArtifact:

        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting Data Ingestion")
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e,sys)
        
    def start_data_validation(self):

        try:
            pass

        except Exception as e:
            raise SensorException(e,sys)
        
    def start_data_transformation(self):

        try:
            pass
        except Exception as e:
            raise SensorException(e,sys)
        
    def start_model_trainer(self):

        try:
            pass
        except Exception as e:
            raise SensorException(e,sys)
        
    def start_model_evaluation(self):

        try:
            pass
        except Exception as e:
            raise SensorException(e,sys)
        
    def start_model_pusher(self):

        try:
            pass
        except Exception as e:
            raise SensorException(e,sys)
    
    def run_pipeline(self):

        try:
            data_ingestion_artifact:DataIngestionConfig = self.start_data_ingestion()
        except Exception as e:
            raise SensorException(e,sys)
    