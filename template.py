import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')
project_name = "transformer_from_scratch"

# List of folders and files 
list_of_files = [
    f"{project_name}/README.md",
    f"{project_name}/requirements.txt",
    f"{project_name}/train.py",
    f"{project_name}/generate.py",
    f"{project_name}/data/.gitkeep",
    f"{project_name}/model/__init__.py",
    f"{project_name}/model/attention.py",
    f"{project_name}/model/transformer_block.py",
    f"{project_name}/model/gpt.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/dataset.py",
    f"{project_name}/venv/.gitkeep"  
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"📁 Created directory: {filedir}")

    if not filepath.exists():
        with open(filepath, "w") as f:
            pass  
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")
