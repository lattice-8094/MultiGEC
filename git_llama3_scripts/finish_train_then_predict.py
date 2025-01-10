import subprocess
import os
import json
from datasets import Dataset
import argparse

"""
This script allows to train a number of models 
(one model per corpora for which a json file is present in "my_train_data_json) and
make predictions with the model on either the dev or the test corpus subsenquently.

It takes one argument: "mode" that can take the values of "dev" or "test".
"""


parser = argparse.ArgumentParser(description="transformation des fichiers dev ou test en format json")
parser.add_argument("mode", type=str, help="Can be train, test or dev")
parser.add_argument("path_to_llama", type=str, help="The path to the llama model.")
args = parser.parse_args()

def run_script_training(script_name, path_to_llama):
    try:
        print(f"Exécution de {script_name}")
        # Exécuter le script avec les arguments lang et steps
        result = subprocess.run(
            ['python', script_name, "./my_train_data_json/", path_to_llama],
            check=True, capture_output=True, text=True
        )
        print(f"{script_name} terminé avec succès.")
        print("Sortie standard :", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de {script_name}: {e.stderr}")

def run_script_prediction(script_name, mode):
    try:
        print(f"Exécution de {script_name} avec mode={mode}")
        # Exécuter le script avec les arguments lang et steps
        result = subprocess.run(
            ['python', script_name, mode],
            check=True, capture_output=True, text=True
        )
        print(f"{script_name} terminé avec succès.")
        print("Sortie standard :", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de {script_name}: {e.stderr}")


run_script_training("train_my_multigec_llms.py", args.path_to_llama)
run_script_prediction("predict_with_my_multigec_llms.py", args.mode)