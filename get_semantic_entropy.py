import os
import json
import torch
import datasets
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
from utils import get_llama_activations_bau, get_llama_activations_bau_custom, get_token_tags
from utils import tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q, tokenized_nq, tokenized_mi, tokenized_from_file, tokenized_from_file_v2
import llama
import pickle
import argparse
from transformers import BitsAndBytesConfig, GenerationConfig, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from peft.tuners.lora import LoraLayer

HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'hl_llama_7B': 'huggyllama/llama-7b',
    'llama_2_7B': 'meta-llama/Llama-2-7b-hf',
    'honest_llama_7B': 'validation/results_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
    'llama_13B': 'huggyllama/llama-13b',
    'llama_30B': 'huggyllama/llama-30b',
    'flan_33B': 'timdettmers/qlora-flan-33b'
}

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--model_cache_dir", type=str, default=None, help='local directory with model cache')
    parser.add_argument("--file_name", type=str, default=None, help='local directory with dataset')
    parser.add_argument('--save_path',type=str, default='')
    args = parser.parse_args()

    print('Loading model responses..')
    prompts, responses = [], []
    if 'baseline' in args.file_name:
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.file_name}.json', 'r') as read_file:
            data = json.load(read_file)
        for i in range(len(data['full_input_text'])):
            prompts.append(data['full_input_text'][i])
            response = data['model_completion'][i] if 'strqa' in args.dataset_name else data['model_answer'][i] # For strqa, we want full COT response
            responses.append(response)
    else:
        # data = []
        with open(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{args.file_name}.json', 'r') as read_file:
            for line in read_file:
                # data.append(json.loads(line))
                data = json.loads(line)
                prompts.append(data['prompt'])
                samples = []
                for i in range(1,args.num_samples+1,1):
                    samples.append(data['response'+str(i)])
                responses.append(samples)
    
    print('Loading deberta..')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()
    
    result_dict = {}
    deberta_predictions = []
    all_semantic_set_ids = []

    print('Calculating semantic similarities...')
    for id_,(question,generated_texts) in enumerate(tqdm(zip(prompts[:10],responses[:10]))):
        unique_generated_texts = list(set(generated_texts))

        answer_list_1 = []
        answer_list_2 = []
        has_semantically_different_answers = False
        inputs = []

        semantic_set_ids = {}
        for index, answer in enumerate(unique_generated_texts):
            semantic_set_ids[answer] = index

        if len(unique_generated_texts) > 1:

            # Evalauate semantic similarity
            for i, reference_answer in enumerate(unique_generated_texts):
                for j in range(i + 1, len(unique_generated_texts)):

                    answer_list_1.append(unique_generated_texts[i])
                    answer_list_2.append(unique_generated_texts[j])

                    qa_1 = question + ' ' + unique_generated_texts[i]
                    qa_2 = question + ' ' + unique_generated_texts[j]

                    input = qa_1 + ' [SEP] ' + qa_2
                    inputs.append(input)
                    encoded_input = tokenizer.encode(input, padding=True)
                    prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)

                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
                    reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                    deberta_prediction = 1
                    # print(qa_1, qa_2, predicted_label, reverse_predicted_label)
                    if 0 in predicted_label or 0 in reverse_predicted_label:
                        has_semantically_different_answers = True
                        deberta_prediction = 0

                    else:
                        semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]

                    deberta_predictions.append([unique_generated_texts[i], unique_generated_texts[j], deberta_prediction])    

        result_dict[id_] = {
            'has_semantically_different_answers': has_semantically_different_answers
        }
        list_of_semantic_set_ids = [semantic_set_ids[x] for x in generated_texts]
        result_dict[id_]['semantic_set_ids'] = list_of_semantic_set_ids
        all_semantic_set_ids.append(list_of_semantic_set_ids)

    print('Saving semantic sets...')
    with open(f'{args.save_path}/uncertainty/{args.model_name}_{args.dataset_name}_{args.file_name}_semantic_similarities.pkl', 'wb') as outfile:
        pickle.dump(result_dict, outfile)
    
    print('Loading sequence predictive entropies...')
    uncertainties = torch.from_numpy(np.load(f'{args.save_path}/uncertainty/{args.model_name}_{args.dataset_name}_{args.file_name}_uncertainty_scores.npy'))
    avg_nll = torch.squeeze(uncertainties[:,:,1]) # uncertainties: (prompts, samples, 2)

    print('Calculating semantic entropies...')
    entropies = []
    for row_index in range(avg_nll[:10].shape[0]):
        aggregated_likelihoods = []
        row = avg_nll[row_index]
        semantic_set_ids_row = torch.Tensor(all_semantic_set_ids[row_index])
        for semantic_set_id in torch.unique(semantic_set_ids_row):
            aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
        aggregated_likelihoods = torch.tensor(aggregated_likelihoods) # - llh_shift
        entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
        entropies.append(entropy.numpy())
    
    print('Saving semantic entropies...')
    np.save(f'{args.save_path}/uncertainty/{args.model_name}_{args.dataset_name}_{args.file_name}_semantic_entropy_scores.npy', entropies)

    print('Estimating SE labels...')
    # First estimate optimal threshold for binarizing
    try_thresholds = np.histogram_bin_edges(entropies[~np.isnan(entropies)], bins='auto')
    objective_func_vals = []
    for threshold in try_thresholds:
        entropies_below_t, entropies_above_t = [], []
        for sem_entropy in entropies:
            if sem_entropy<threshold:
                entropies_below_t.append(sem_entropy)
            else:
                entropies_above_t.append(sem_entropy)
        low_entropy_avg, high_entropy_avg = np.mean(entropies_below_t), np.mean(entropies_above_t)
        low_entropy_sum_sq_err, high_entropy_sum_sq_err = np.sum([(ent-low_entropy_avg)**2 for ent in entropies_below_t]), np.sum([(ent-high_entropy_avg)**2 for ent in entropies_above_t])
        objective_func_vals.append(low_entropy_sum_sq_err + high_entropy_sum_sq_err)
    optimal_threshold = try_thresholds[np.argmin(objective_func_vals)]
    # Labels
    se_labels = [1 if ent>optimal_threshold else 0 for ent in entropies]

    print('Saving semantic entropy labels...')
    np.save(f'{args.save_path}/uncertainty/{args.model_name}_{args.dataset_name}_{args.file_name}_se_labels.npy', se_labels)


if __name__ == '__main__':
    main()