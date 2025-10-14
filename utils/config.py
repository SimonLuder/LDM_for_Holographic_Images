import yaml
import argparse

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)
    

def update_config(config:dict, args: dict):
    config.update(args)

def save_config(config, config_file):
    with open(config_file, 'w') as stream:
        try:
            yaml.safe_dump(config, stream)
        except yaml.YAMLError as exc:
            print(exc)

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value

