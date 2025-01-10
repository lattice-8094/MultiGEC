from trl import setup_chat_format
import os
from transformers import TrainingArguments 
# disable Weights and Biases
os.environ['WANDB_DISABLED']="true"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from peft import AutoPeftModelForCausalLM
from transformers import pipeline
import json
from datasets import Dataset
import pickle



parser = argparse.ArgumentParser(description="Use a fine-tunedmy_ Llama 8B for grammatical error correction.")
parser.add_argument("corpus_name", type=str, help="The corpus_name of the essays.")
parser.add_argument("mode", type=str, help="Can be test or dev")
parser.add_argument("file", type=str, help="path to the file containing the json of stuff that needs prediction.")
parser.add_argument("output_dir", type=str, help="path to the directory where outputs need to be stored")
parser.add_argument("max_tokens", type=int, help="maximum tokens to generate")



args = parser.parse_args()

path_to_model = "./my_finetuned_models/my_"+args.corpus_name+"_finetuned_model"


peft_model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_model,
    device_map="auto",
    torch_dtype=torch.float16
    )
tokenizer = AutoTokenizer.from_pretrained(path_to_model)
pipe = pipeline("text-generation", model=peft_model, tokenizer=tokenizer)      

def extraire_nom_fichier(chemin):
    """
    Extrait la partie spécifique du chemin (par exemple, 'cs-natform-orig-test.md').

    Args:
        chemin (str): Chemin complet du fichier.

    Returns:
        str: Nom de fichier sans la dernière extension JSON.
    """
    # Récupérer le nom du fichier complet
    nom_complet = os.path.basename(chemin)  # Ex: "test_cs-natform-orig-test.md.json"

    # Retirer la dernière extension (".json")
    nom_sans_json = os.path.splitext(nom_complet)[0]  # Ex: "test_cs-natform-orig-test.md"

    # Supprimer le préfixe indésirable ("test_")
    if nom_sans_json.startswith("test_"):
        nom_sans_json = nom_sans_json[len("test_"):]

    return nom_sans_json
        

system_message = """You are a grammatical error correction tool. Your task is to correct the grammaticality and spelling of the input essay written by a learner. Return only the corrected text and nothing more."
{schema}"""

#Importing the dataset
#dataset_name = "/data/shared-tasks/mgec2025/fine_tuning_Llama3/dev_data_german.json"


with open(args.file) as json_file:
    data = json.load(json_file)

dataset = Dataset.from_dict(data)


data_hf_format = tokenizer.apply_chat_template(dataset['data'], tokenize=False)#['text'][3]
dataset = dataset.add_column("text", data_hf_format)
#dev_dataset

print("size of the dataset", dataset)
#input()


my_outputs = []


for i in range(len(dataset)):

    prompt = dataset[i]['text']+"<|start_header_id|>assistant<|end_header_id|>"

    outputs = pipe(prompt, max_new_tokens=int(args.max_tokens), do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

    print("item number", i + 1)
    print("number of items in total", len(dataset))
    print("\n\n", flush=True)
    

    my_outputs.append(outputs)

    import ast

def transform_and_save(outputs, essay_ids, output_filename):
    """
    Transforme les outputs en extrayant le texte transformé et génère un fichier .md.
    
    Args:
        outputs (list of str): Liste des strings représentant les outputs à transformer.
        essay_ids (list of str): Liste des identifiants des essais correspondants.
        output_filename (str): Nom du fichier de sortie.
    """
    if len(outputs) != len(essay_ids):
        raise ValueError("La liste des outputs et la liste des identifiants doivent avoir la même longueur.")
    
    with open(output_filename, "w") as my_file:
        for output, essay_id in zip(outputs, essay_ids):
            system_answer = ""
            system_answer = output[0]['generated_text'].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
            system_answer = system_answer.split("'}]")[0]
            my_file.write(f"### essay_id = {essay_id}\n")
            my_file.write(f"{system_answer}\n\n")
            

# open a file, where you stored the pickled data
file_essay_ids = open(os.path.join('./my_essay_ids', 'my_essai_ids_'+args.mode+'_' + extraire_nom_fichier(args.file)), 'rb')


# dump information to that file
my_essai_ids = pickle.load(file_essay_ids)

# close the file
file_essay_ids.close()


file_my_outputs = open(os.path.join(os.path.join('./', args.output_dir), extraire_nom_fichier(args.file).replace('orig', 'hypo')), 'wb')

pickle.dump(my_outputs, file_my_outputs)

file_my_outputs.close()

output_filename = os.path.join(os.path.join('./', args.output_dir + '_md_format/'), extraire_nom_fichier(args.file).replace('orig', 'hypo'))


transform_and_save(my_outputs, my_essai_ids, output_filename)