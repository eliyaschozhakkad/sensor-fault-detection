from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity.artifact_entity import DataValidationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from sensor.entity.config_entity import ModelEvaluationConfig,TrainingPipelineConfig
import os,sys
from sensor.ml.metric.classfication_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel,ModelResolver
from sensor.utils.main_utils import save_object,load_object,write_yaml_file
import  pandas as pd
from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.ml.model.estimator import TargetValueMapping


class ModelEvaluation:

    def __init__(self,model_eval_config:ModelEvaluationConfig,
                 data_validation_artifact:DataValidationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise SensorException(e,sys)
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        try:
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            #valid train and test dataframe
            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

            logging.info("Obtained train and test dataframe")

            df = pd.concat([train_df,test_df])

            train_model_file_path = self.model_trainer_artifact.trained_model_file_path

            logging.info("Calling Model Resolver")
            model_resolver = ModelResolver()

            is_model_accepted = True
            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted = is_model_accepted,
                    improved_accuracy = None,
                    best_model_path = None,
                    trained_model_path = train_model_file_path,
                    train_model_metric_artifact = self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact = None )
                logging.info(f"Model evaluation artifact :{model_evaluation_artifact}")
                
                return model_evaluation_artifact
            
            logging.info("Loading latest and best model")
            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path = latest_model_path)
            logging.info(f"latest_model:{latest_model}")
            train_model = load_object(file_path = train_model_file_path)
            logging.info(f"train_model:{train_model}")

            y_true = df[TARGET_COLUMN]
            y_true = y_true.replace(TargetValueMapping().to_dict())
            
            df = df.drop(columns=[TARGET_COLUMN],axis =1)
            y_trained_pred = train_model.predict(df)
            logging.info(f"y_trained_pred:{y_trained_pred}")
            y_latest_pred = latest_model.predict(df)
            logging.info(f"y_latest_pred:{y_latest_pred}")

            logging.info("Calculating classification metrics for latest and best model")
            trained_metric = get_classification_score(y_true, y_trained_pred)
            latest_metric = get_classification_score(y_true, y_latest_pred)

            
            improved_accuracy = latest_metric.f1_score - trained_metric.f1_score
            logging.info(f"latest_metric.f1_score :{latest_metric.f1_score}")
            logging.info(f"trained_metric.f1_score : {trained_metric.f1_score}")
            logging.info(f"Improved accuracy: {improved_accuracy}")

            if self.model_eval_config.change_threshold < improved_accuracy:
                is_model_accepted = True
                logging.info("MOdel is accepted")
            else:
                is_model_accepted = False
                logging.info("MOdel is not accepted")

            model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted = is_model_accepted,
                    improved_accuracy = improved_accuracy,
                    best_model_path = latest_model_path,
                    trained_model_path = train_model_file_path,
                    train_model_metric_artifact = trained_metric,
                    best_model_metric_artifact = latest_metric )
            
            model_eval_report = model_evaluation_artifact.__dict__

            #Save report
            logging.info("Saving model evaluation report")
            write_yaml_file(self.model_eval_config.report_file_name, model_eval_report)

            logging.info(f"Model evaluation artifact :{model_evaluation_artifact}")   
            return model_evaluation_artifact

        except Exception as e:
            raise SensorException(e,sys) 