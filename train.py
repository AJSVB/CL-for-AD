import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="train.yaml")
def main(config: DictConfig,exp_number):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.training_pipeline import train

    # Applies optional utilities
    utils.extras(config)
    if exp_number != None:
        config['datamodule']['label_class'] = exp_number
    # Train model
    return train(config)


if __name__ == "__main__":
    import os
    import sys
    if(len(sys.argv)>1):
        i = sys.argv[1]
    else:
        i = None
    os.chdir("./data/diagvibsix/")
    os.system("source activate diagvibsix")
    os.system("pip install -e .")
    os.chdir("./../..")
    main(i)
