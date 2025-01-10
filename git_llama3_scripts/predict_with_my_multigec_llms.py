import subprocess
from pathlib import Path
import os
import argparse



parser = argparse.ArgumentParser(description="transformation des fichiers dev ou test en format json")
parser.add_argument("mode", type=str, help="Can be test or dev")
args = parser.parse_args()



if not os.path.exists("my_generalized_inference_script.py"):
    raise FileNotFoundError("Did not find my_generalized_inference_script.py")

#pour passer tous les fichiers test
repertoire = Path("./my_"+args.mode+"_data_json")



def run_script(script_name, corpus_name, mode, file, output_dir, max_tokens):
    try:
        print(f"Exécution de {script_name} avec corpus_name={corpus_name} et mode={mode} et file={file} et output_dir: {output_dir} et max_tokens : {max_tokens}...")
        # Exécuter le script avec les arguments corpus_name et steps
        print(script_name, corpus_name, mode, file, output_dir, max_tokens)
        result = subprocess.run(
            ['python', script_name, corpus_name, mode, file, output_dir, str(max_tokens)],
            check=True, capture_output=True, text=True
        )
        print(f"{script_name} terminé avec succès.")
        print("Sortie standard :", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de {script_name}: {e.stderr}")


# Spécifiez le chemin du répertoire

get_max_tokens = {"cs-natform" : 1323, 
                  "cs-natwebinf" :422,
                  "cs-romani" : 909,
                  "cs-seclearn" : 1459,
                  "en-writeandimprove2024" : 604,
                  "et-eic" : 882,
                  "et-ekil2" : 643,
                  "de-merlin" : 440,
                  "el-glcii" : 952,
                  "is-iceec" : 4221,
                  "is-icel2ec" : 3070,
                  "it-merlin" : 478,
                  "lv-lava" : 1152,
                  "ru-rulec" : 2933,
                  "sl-solar_eval" : 3178,
                  "sv-swell_gold" : 1319,
                  "uk-ua_gec" : 1905, 
                  "dummy-dummy": 100}


# Boucler sur tous les fichiers dans le répertoire
for fichier in repertoire.iterdir():
    if fichier.is_file():  # Vérifie que c'est un fichier
        print(fichier)
        list_name_corpus = (os.path.basename(fichier).split("-")[:2])
        corpus_name = '-'.join(list_name_corpus)
        #corpus_name = str(fichier).split("-")[:2]
        print(corpus_name)
        mode = args.mode
        file = fichier
        output_dir = "my_predicted_"+mode+"_outputs"
        max_tokens =  get_max_tokens[corpus_name]+int(0.15*get_max_tokens[corpus_name])
        if max_tokens > 8192 :
            max_tokens = 8192
        #print("launchin my_generalized inference script with :", corpus_name, mode, file, output_dir)
        run_script("my_generalized_inference_script.py", corpus_name, mode, file, output_dir, max_tokens)