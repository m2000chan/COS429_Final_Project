import importlib.util
import os

def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if __name__ == '__main__':
    gcnn_vs_cnn_module = import_from_file("gcnn_vs_cnn_testing", os.path.join("GCNN vs CNN", "testing.py"))
    gcnn_vs_cnn_module.train()

    gcnn_vs_cnn_da_module = import_from_file("gcnn_vs_cnn_da_testing", os.path.join("GCNN vs CNN with Data Augmentation", "testing.py"))
    gcnn_vs_cnn_da_module.train_with_DA()
