'''
Created on 22 nov. 2024

@author: olga
'''

import subprocess
import os
import json
from datasets import Dataset
import argparse

parser = argparse.ArgumentParser(description="train llama on the multigec datasets")
parser.add_argument("path_to_directory", type=str, help="point to directory with scripts and directories with data")
parser.add_argument("path_to_llama", type=str, help="The path to the llama model.")
args = parser.parse_args()


if not os.path.exists("fine_tune_script.py"):
    raise FileNotFoundError("Did not find fine_tune_script.py")

def run_script(script_name, corpus_name, steps, path_to_llama):
    try:
        print(f"Exécution de {script_name} avec corpus_name={corpus_name} et steps={steps}...")
        # Exécuter le script avec les arguments lang et steps
        result = subprocess.run(
            ['python', script_name, corpus_name, str(steps), str(path_to_llama)],
            check=True, capture_output=True, text=True
        )
        print(f"{script_name} terminé avec succès.")
        print("Sortie standard :", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de {script_name}: {e.stderr}")


scripts = []

def count_essay_id_in_file(file_path):
        with open(file_path) as json_file:
            data = json.load(json_file)
        dataset = Dataset.from_dict(data)
        return len(dataset["data"])



for file_name in os.listdir(args.path_to_directory):
    file_path = os.path.join(args.path_to_directory, file_name)
    # Vérifie si c'est un fichier
    corpus_name = file_name.removesuffix("-orig-full.md.json")
    print(corpus_name)
    if os.path.isfile(file_path):
        count = count_essay_id_in_file(file_path)
        print(count)
        if count < 100 : 
            steps = 50
        else :
            steps = 150
        run_script("fine_tune_script.py", corpus_name, steps, args.path_to_llama)
