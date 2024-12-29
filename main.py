from cnnClassifier import logger
from cnnClassifier.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_2_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_3_model_trainer import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_4_evaluation import EvaluationPipeline


STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>Stage {STAGE_NAME} started <<<<<<<<<<")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare base model"

try:
    logger.info(f">>>>>Stage {STAGE_NAME} started <<<<<<<<<<")
    prepare_base_model=PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training"


try:
    logger.info(f">>>>>Stage {STAGE_NAME} started <<<<<<<<<<")
    model_trainer=ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation"
try:
    logger.info(f">>>>>Stage {STAGE_NAME} started <<<<<<<<<<")
    eval=EvaluationPipeline()
    eval.main()
    logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e