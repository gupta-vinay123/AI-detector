from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.evaluation import Evaluation
from cnnClassifier import logger

STAGE_NAME = "Evaluation"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        #evaluation.log_into_mlflow()


if __name__ =="__main__":
    try:
        logger.info(f">>>>>Stage {STAGE_NAME} started <<<<<<<<<<")
        eval=EvaluationPipeline()
        eval.main()
        logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e