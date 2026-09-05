from pathlib import Path
from datasets import Dataset, load_dataset, concatenate_datasets
from config import load_config

cfg = load_config()

def load_datasets_from_config(config_datsets) -> Dataset:
    datasets = []
    for ds in config_datsets:
        dataset = load_dataset(ds.source, ds.name, split=ds.split)
        dataset = dataset.rename_columns(ds.columns_map)
        target_columns = list(ds.columns_map.values())
        dataset = dataset.select_columns(target_columns)
        datasets.append(dataset) 
    return concatenate_datasets(datasets)

def load_ru_latex_datasets() -> tuple[Dataset, Dataset]:
    ru_dataset = load_datasets_from_config(cfg.ru_datasets, ru_datasets)
    latex_dataset = load_datasets_from_config(cfg.latex_datasets, latex_datasets)
    return ru_dataset, latex_dataset

        
