# Disclaimer

This code was developed by Olga Seminck. She used ChatGPT for the development. 
She only promted ChatGPT to generate little pieces of code and it never happend that data was send to OpenAI.


# Instructions

## 0. Execute scripts from this directory

This program can only work if users execute scripts from this directory. 

## 1. Transform data to json format

All the original multigec essays need to be converted to json format. 

To do this, use the script "transform_to_json.py"

It takes 2 arguments:

First, the mode that can be "train", "dev" or "test"

Second, it takes the path to the repository including the scripts and data respositories, for example: "/data/shared-tasks/mgec2025/fine_tuning_Llama3_run2/".

```
python transform_to_json.py train "./"
```

## 2. Finetuning

Use the script "train_my_multigec_llms.py" to train one fine-tuned Llama3.0 model per corpus. One model will be trained for each json file present in the respository "my_train_data_json" and saved in the directory "my_finetuned_models".

To execute: call the script with two arguments (the path to the training data in json format and the path where the llama model is saved).

```
python train_my_multigec_llms.py /path/to/my_train_data_json path/to/directory_with_weights_of_llama3
```

Users can update the number of optimization steps in line 49 and 51 of this script.

The script "train_my_multigec_llms.py" calls the script "fine_tune_script.py" but a user does not need to interact with this script directly.

## 3. Prediction

Use the script "predict_with_my_multigec_llms.py" to make predictions on the files in the dev or the test corpus. 

```
predict_with_my_multigec_llms.py test_or_dev 
````

The script "predict_with_my_multigec_llms.py" calls the script "my_generalized_inference_script.py" but a user does not need to interact with this script directly.

## 4. Finetuning and Prediction 

Simply execute the script "finish_train_then_predict.py" It takes two argument: dev or test and the path to the weights of llama. 

```
predict_with_my_multigec_llms.py test_or_dev 
````

It trains models for all the files present in the "my_train_data_json" and makes predictions for all the
files in the folders "my_dev_data_json" or "my_test_data_json" according to the mode that was chosen. 
