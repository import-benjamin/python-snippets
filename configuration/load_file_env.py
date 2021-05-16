def load_configuration(config_file: str = "config.ini"):
    """
    :param: config_file
    :return: dict()
    """

    from configparser import ConfigParser
    from os.path import isfile
    from os import environ, getenv

    config_file_buffer = dict()

    if isfile(config_file):
        config_parser = ConfigParser(environ)
        config_parser.read(config_file)
        config_file_buffer = config_parser["config"]

    # load environment
    environment_keywords = ("CONFIG1", "CONFIG2", "CONFIG3")
    config_env_buffer = {
        key: getenv(key) for key in environment_keywords if getenv(key)
    }

    return {**configuration_buffer, **config_env_buffer}
