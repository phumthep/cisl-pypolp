import configparser
import os



def get_root_dir() -> str:
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.realpath(__file__)))))


def get_config() -> configparser.ConfigParser:
    config_file = os.path.join(get_root_dir(), 'user_config.ini')
    config = configparser.ConfigParser()
    config.read(config_file)
    return config
    
    