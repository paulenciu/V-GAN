import importlib
import os
from pathlib import Path

from src.models.generator.AbstractGenerator import AbstractGenerator


class Instantiator:

    def instantiate_file(self, base_dir, target_name, latent_size):
        base_dir = Path(base_dir)

        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    spec = importlib.util.spec_from_file_location("module.name", file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, AbstractGenerator) and attr is not AbstractGenerator:
                            return attr(latent_size)