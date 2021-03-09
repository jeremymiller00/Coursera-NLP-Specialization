import yaml
from loguru import logger as log
from Autocomplete import Autocomplete


def load_config(path: str) -> dict:
    """
    Load and parse config yaml file
    :param path: string, path to yaml file
    :return: dict of config parameters
    """
    with open(path, 'r') as source:
        config = yaml.safe_load(source)
    return config


def main(config):
    if "log_file" in config.keys():
        log.add(config["log_file"])
    ac = Autocomplete(N = 3)


if __name__ == "__main__":
    config = load_config("config.yaml")
    main(config)
