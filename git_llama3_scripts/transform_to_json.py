'''
Created on 22 nov. 2024

@author: olga
'''
import re
import json
import pickle as pk
import os
import argparse


parser = argparse.ArgumentParser(description="transformation des fichiers dev ou test en format json")
parser.add_argument("mode", type=str, help="Can be train, test or dev")
parser.add_argument("path_to_directory", type=str, help="point to directory with scripts and directories with data")

args = parser.parse_args()

path_on_clothos = args.path_to_directory
path = os.path.join(path_on_clothos, "my_"+args.mode+"_files/")
path_json = os.path.join(path_on_clothos, "my_"+args.mode+"_data_json/")
path_essay_ids = os.path.join(path_on_clothos, "my_essay_ids/")

def extract_essays(file_path):
        essays = []
        my_essay_ids = []

        current_essay = None
    
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()  # Supprimer les espaces inutiles en début/fin de ligne
                if line.startswith("### essay_id ="):
                    # Si un nouvel identifiant est détecté
                    if current_essay:  # Enregistrer l'essai en cours, s'il existe
                        if current_essay["text"][-2:]== "\n\n":
                            current_essay["text"] = current_essay["text"].replace("\n\n", " $$$")
                        essays.append(current_essay)
                    current_essay = {"essay_id": line[len("### essay_id ="):].strip(), "text": ""}
                    my_essay_ids.append(line[len("### essay_id ="):].strip())
                elif current_essay:  # Ajouter le texte à l'essai en cours
                    current_essay["text"] += (line + "\n")  # Conserver les sauts de ligne
    
        # Ajouter le dernier essai, si existant
        if current_essay:
            essays.append(current_essay)

    
        return (essays, my_essay_ids)

prompt = "You are a grammatical error correction tool. Your task is to correct the grammaticality and spelling of the input essay written by a learner. Return only the corrected text and nothing more."


for file_path in os.listdir(path):
    if file_path.find("orig") != -1:
        result = {"data":[]}
        original_file = os.path.join(path, file_path)
        (original_essays, my_essay_ids) = extract_essays(original_file)
        #print("file_path", file_path)
        #input()
        corrected_file = ""
        (corrected_essays, my_essay_ids_corr) = (None, None)
        if args.mode == "train" :
            corrected_file = path  + file_path.replace("orig", "ref")
            (corrected_essays, my_essay_ids_corr) = extract_essays(corrected_file)
            for original, corrected in zip(original_essays, corrected_essays):
                result["data"].append(
                    [{"role": "system", "content": f"{prompt}"},
                    {"role": "user", "content": f"{original['text']}"}, 
                    {"role": "assistant", "content": corrected['text']}]
                )
        else:
            for original in original_essays:

                result["data"].append(
                    [{"role": "system", "content": f"{prompt}"},
                    {"role": "user", "content": f"{original['text']}"}] #, 
                    #{"role": "assistant", "content": corrected['text']}]
                )
                
    
        #print(lang,len(result["data"]))
        
        file_essay_ids = open(path_essay_ids + "my_essai_ids_"+args.mode+"_" + file_path.lower(), 'wb')
        
        # dump information to that file
        pk.dump(my_essay_ids, file_essay_ids)
        
        # close the file
        file_essay_ids.close()
        
        # Écriture dans un fichier JSON
        output_file = path_json + file_path.lower() + ".json"
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(result, file, indent=4, ensure_ascii=False)

            
        print(f"Fichier JSON généré : {output_file}")
