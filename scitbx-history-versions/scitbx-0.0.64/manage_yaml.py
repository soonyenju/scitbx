import yaml
from pathlib import Path

class Yaml(object):
    def __init__(self, path):
        if isinstance(path, Path):
            path = path.as_posix() 
        self.path = path

    def load(self):
        with open(self.path, "r") as stream:
            try:
                yamlfile = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                assert(exc)
        return yamlfile
    
    def dump(self, data_dict):
        with open(self.path, "w") as f:
            yaml.dump(data_dict, f, default_flow_style = False)