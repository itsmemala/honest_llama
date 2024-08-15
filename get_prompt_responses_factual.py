import os
import torch
import datasets
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter
import numpy as np
import pickle
import string
import re
import json
import jsonlines
import random
import llama
import argparse
# from transformers import BitsAndBytesConfig, GenerationConfig
# from peft import PeftModel
# from peft.tuners.lora import LoraLayer
import evaluate

import ssl
import urllib.request
import zipfile

# Squad F1 calculation from: https://github.com/tangbinh/question-answering/blob/master/evaluate.py

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def my_squad_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 6
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "So the answer is"
SHORT_ANSWER_TRIGGER = "answer is" # for long answer

def load_jsonl(file_path, is_gzip=False):
    # Format of each line in StrategyQA:
    # {"qid": ..., "term": ..., "description": ..., "question": ..., "answer": ..., "facts": [...], "decomposition": [...]}
    
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        items = json.load(f)
        for item in items:
            new_item = dict(
                qid=item.get('qid', None),
                # term=item.get('term', None),
                # description=item.get('description', None),
                question=item.get('question', None),
                answer=item.get('answer', None),
                # facts=item.get('facts', []),
                # decomposition=item.get('decomposition', [])
            )
            list_data_dict.append(new_item)
    return list_data_dict

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def create_demo_text(n_shot=6, cot_flag=True, shuffle=False):
    question, chain, answer = [], [], []
    question.append("Do hamsters provide food for any animals?")
    chain.append("Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.")
    answer.append("yes")

    question.append("Could Brooke Shields succeed at University of Pennsylvania?")
    chain.append("Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.")
    answer.append("yes")

    question.append("Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?")
    chain.append("Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5.")
    answer.append("no")

    question.append("Yes or no: Is it common to see frost during some college commencements?")
    chain.append("College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.")
    answer.append("yes")

    question.append("Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?")
    chain.append("The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.")
    answer.append("no")

    question.append("Yes or no: Would a pear sink in water?")
    chain.append("The density of a pear is about 0.6 g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float.")
    answer.append("no")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    if shuffle:
        random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text


def build_prompt(input_text, n_shot, cot_flag, shuffle, dataset_name):
    demo = create_demo_text(n_shot, cot_flag, shuffle) if dataset_name=='strqa' else create_demo_text_gsm8k(n_shot, cot_flag, shuffle)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def clean_answer(model_pred, random_guess=False):
    model_pred = model_pred.lower()

    if "Thus, yes." in model_pred:
        preds = "yes"
    elif SHORT_ANSWER_TRIGGER.lower() in model_pred:
        preds = model_pred.split(SHORT_ANSWER_TRIGGER.lower())[1].split(".")[0].strip()
    else:
        print("Warning: answer trigger not found in model prediction:", model_pred, "; returning yes/no based on exact match of `no`.", flush=True)
        if random_guess:
            preds = "no" if "no" in model_pred else "yes"
        else:
            return None
    if preds not in ["yes", "no"]:
        print("Warning: model prediction is not yes/no:", preds, "; returning no", flush=True)
        if random_guess:
            preds = "no"
        else:
            return None

    return (preds == "yes")

def load_jsonl_gsm8k(file_path,
               instruction='instruction',
               input='input',
               output='output',
               category='category',
               is_gzip=False):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None)
            item = new_item
            list_data_dict.append(item)
    return list_data_dict

def is_correct_gsm8k(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def create_demo_text_gsm8k(n_shot=8, cot_flag=True, shuffle=False):
    question, chain, answer = [], [], []
    question.append("There are 15 trees in the grove. "
                    "Grove workers will plant trees in the grove today. "
                    "After they are done, there will be 21 trees. "
                    "How many trees did the grove workers plant today?")
    chain.append("There are 15 trees originally. "
                 "Then there were 21 trees after some more were planted. "
                 "So there must have been 21 - 15 = 6.")
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?")
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?")
    chain.append("Originally, Leah had 32 chocolates. "
                 "Her sister had 42. So in total they had 32 + 42 = 74. "
                 "After eating 35, they had 74 - 35 = 39.")
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?")
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8.")
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?")
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9.")
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?")
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29.")
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?")
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls.")
    answer.append("33")

    question.append("Olivia has $23. She bought five bagels for $3 each. "
                    "How much money does she have left?")
    chain.append("Olivia had 23 dollars. "
                 "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
                 "So she has 23 - 15 dollars left. 23 - 15 is 8.")
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    if shuffle:
        random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text

def clean_answer_gsm8k(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

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
    'flan_33B': 'timdettmers/qlora-flan-33b'
}

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='nq_open', help='dataset for querying the model')
    parser.add_argument('--len_dataset', type=int, default=0)
    parser.add_argument('--start_at', type=int, default=0)
    parser.add_argument('--use_split', type=str, default='validation')
    parser.add_argument('--hallu_check_prompt', type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--num_ret_seq', type=int, default=1)
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument('--device', type=int, default=0)    
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--model_cache_dir", type=str, default=None, help='local directory with model cache')
    parser.add_argument('--save_path',type=str, default='')
    parser.add_argument('--seed',type=int, default=42)
    args = parser.parse_args()

    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print('Loading model..')
    tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL)
    # if args.num_ret_seq>1 and args.model_name=='alpaca_7B': os.environ["PYTORCH_USE_CUDA_DSA"] = "1" #tokenizer.pad_token = tokenizer.eos_token
    model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    if args.num_ret_seq>1 and args.model_name=='llama_2_7B': model = model.bfloat16() # Numerical instability; Solution from: https://github.com/meta-llama/llama/issues/380
    device = "cuda"
    # device = 'cpu' # for debugging
    # model = model.cpu()

    print('Loading data..')
    print(args.len_dataset,args.start_at,args.use_split)
    # Load data
    len_dataset = args.len_dataset
    start_at = args.start_at
    if args.dataset_name=='strqa':
        fp = os.path.join(args.save_path, 'strategyqa_train.json')
        if not os.path.exists(fp):
            download_url(
                'https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip', args.save_path)
            # Once the file is downloaded, unzip it
            with zipfile.ZipFile(os.path.join(args.save_path, 'strategyqa_dataset.zip'), 'r') as zip_ref:
                zip_ref.extractall(args.save_path)
        list_data_dict = load_jsonl(fp)
        all_input_texts, all_gt_answers, tokenized_prompts = [], [], []
        for sample in list_data_dict:
            all_gt_answers.append(sample['answer'])
            input_text = build_prompt(sample['question'], N_SHOT, COT_FLAG, args.do_shuffle, args.dataset_name)
            all_input_texts.append(input_text)
            tokenized_prompt = tokenizer(input_text, return_tensors = 'pt').input_ids
            tokenized_prompts.append(tokenized_prompt)
    elif args.dataset_name=='gsm8k':
        fp = os.path.join(args.save_path, 'gsm8k_'+args.use_split+'.json')
        download_path = 'https://raw.githubusercontent.com/openai/'\
                +'grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/'\
                +'grade_school_math/data/'+args.use_split+'.jsonl'
        if not os.path.exists(fp):
            download_url(download_path, args.save_path)
            os.rename(args.save_path+'/'+args.use_split+'.jsonl', fp)
        list_data_dict = load_jsonl_gsm8k(fp, instruction='question', output='answer')
        all_input_texts, all_gt_answers, tokenized_prompts = [], [], []
        for sample in list_data_dict[:args.len_dataset]:
            all_gt_answers.append(sample['output'])
            input_text = build_prompt(sample['instruction'], N_SHOT, COT_FLAG, args.do_shuffle, args.dataset_name)
            all_input_texts.append(input_text)
            tokenized_prompt = tokenizer(input_text, return_tensors = 'pt').input_ids
            tokenized_prompts.append(tokenized_prompt)
    else:
        if args.dataset_name=='nq_open':
            hf_dataset_name = 'nq_open'
            dataset = load_dataset(hf_dataset_name, streaming= True)[args.use_split]
        elif args.dataset_name=='trivia_qa':
            hf_dataset_name = 'mandarjoshi/trivia_qa'
            dataset = load_dataset(hf_dataset_name, 'rc.nocontext', streaming= True)[args.use_split]
        elif args.dataset_name=='cnn_dailymail':
            hf_dataset_name = 'cnn_dailymail'
            dataset = load_dataset(hf_dataset_name, streaming= True)[args.use_split]
        if args.hallu_check_prompt is None:
            prompts = []
            tokenized_prompts = []
            for idx,val in enumerate(list(dataset.take(len_dataset))[start_at:]):
                if args.dataset_name=='nq_open':
                    question = val['question']
                    cur_prompt = f"This is a bot that correctly answers questions. \n Q: {question} A: "
                elif args.dataset_name=='trivia_qa':
                    question = val['question']
                    cur_prompt = f"This is a bot that correctly answers questions. \n Q: {question} A: "
                elif args.dataset_name=='cnn_dailymail':
                    article = val['article']
                    cur_prompt = f"Article: {article}\n Summarize the article in two to three sentences. Summary: "
                prompts.append(cur_prompt)
                tokenized_prompt = tokenizer(cur_prompt, return_tensors = 'pt').input_ids
                tokenized_prompts.append(tokenized_prompt)
        else:
            # Load greedy responses
            greedy_resp_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_greedy_responses_{args.use_split}{args.len_dataset}.json'
            with open(greedy_resp_fname, 'r') as read_file:
                greedy_resp_data = []
                for line in read_file:
                    greedy_resp_data.append(json.loads(line))
            prompts = []
            tokenized_prompts = []
            for row,val in zip(greedy_resp_data,list(dataset.take(len_dataset))[start_at:]):
                if args.hallu_check_prompt==1:
                    cur_prompt = row['prompt'] + row['response1'] + "\n The above generated answer is incorrect. Revised answer: "
                if args.hallu_check_prompt==2:
                    cur_prompt = row['prompt'] + row['response1'] + "\n The above answer may be incorrect. The actual correct answer is: "
                if args.hallu_check_prompt==3 and (args.dataset_name=='trivia_qa' or args.dataset_name=='nq_open'):
                    question = val['question']
                    cur_prompt = f"This is a bot that correctly answers questions. Consider the below question and a possible answer, which may or may not be correct. Provide the correct answer to the question. \n Q: {question} Possible answer: {row['response1']}\n Correct answer:"
                prompts.append(cur_prompt)
                tokenized_prompt = tokenizer(cur_prompt, return_tensors = 'pt').input_ids
                tokenized_prompts.append(tokenized_prompt)
    
    print('Getting model responses..')
    # Get model responses
    responses = []
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []} #, 'raw_model_generation': []}
    correct_rate = 0
    oom_err_idxs = []
    if args.dataset_name=='strqa':
        # period_token_id = tokenizer("\n\n##")['input_ids'] # is this ok?
        period_token_id = tokenizer("\n")['input_ids'] # is this ok?
        eos_tokens = ["Q:", "\n\n##"]
        checkgens = ["Q:", "\n\n##"]
    if args.dataset_name=='gsm8k':
        period_token_id = tokenizer(".")['input_ids']
        eos_tokens = ["Q:", "\end{code}"] #, "\n\nQ"]
        checkgens = ["Q:", "\end{code}"] #, "\n\nQ"]
    elif args.dataset_name=='nq_open' or args.dataset_name=='trivia_qa':
        # period_token_id = tokenizer('. ')['input_ids'][1]
        period_token_id = tokenizer('.')['input_ids']
        eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:', ' Q:', 'A:', ' A:',
                        'QA:', ' QA:', 'QA1', ' QA1', '.\n', ' \n', ':', "\\"]
        # question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in eos_tokens]
        # eos_tokens = ['Question', '\n', 'Answer', ':']
        checkgens = ['QA2:','Q.', 'B:']
    elif args.dataset_name=='cnn_dailymail':
        period_token_id = tokenizer('\n')['input_ids']
        eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:', ' Q:', 'A:', ' A:',
                        'QA:', ' QA:', 'QA1', ' QA1', '.\n', ' \n', ':', "\\", 'Summary:', ' Summary:']
        checkgens = ['Summary:']
    question_framing_ids = [tokenizer(eos_token, add_special_tokens=False)['input_ids'] for eos_token in eos_tokens]
    # print('Bad word ids:',question_framing_ids)
    for i,tokenized_prompt in enumerate(tqdm(tokenized_prompts[:1])):
        print(prompt[i])
        tokenized_prompt = tokenized_prompt.to(device)
        # try:
            # response = model.generate(tokenized_prompt, max_new_tokens=512,
            #                             # num_beams=1,
            #                             temperature=args.temperature, top_p=args.top_p, do_sample=args.do_sample, num_return_sequences=args.num_ret_seq,
            #                             eos_token_id=period_token_id,
            #                             bad_words_ids=question_framing_ids + [tokenized_prompt.tolist()[0]]
            #                             )[:, tokenized_prompt.shape[-1]:]
        response = []
        for j in range(args.num_ret_seq):
            response.append(model.generate(tokenized_prompt, max_new_tokens=512,
                                    # num_beams=1,
                                    temperature=args.temperature, top_p=args.top_p, do_sample=args.do_sample, num_return_sequences=1,
                                    eos_token_id=period_token_id,
                                    bad_words_ids=question_framing_ids + [tokenized_prompt.tolist()[0]]
                                    )[:, tokenized_prompt.shape[-1]:])
        # except torch.cuda.OutOfMemoryError: # This is for strqa sampling: Skip samples that don't fit on gpu
        #     is_cor, model_answer, model_completion, input_text = [], [], [], []
        #     result_dict['is_correct'].append(is_cor)
        #     result_dict['model_answer'].append(model_answer)
        #     result_dict['model_completion'].append(model_completion)
        #     result_dict['full_input_text'].append(input_text)
        #     oom_err_idxs.append(i)
        #     continue
        if args.num_ret_seq==1:
            if args.dataset_name=='strqa' or args.dataset_name=='gsm8k':
                cur_response = tokenizer.decode(response[0][0], skip_special_tokens=True) # Note: [0] only needed because of temp fix to loop through num_ret_seq
                for check_gen in checkgens: # Fix generation stopping errors
                    cur_response = cur_response.split(check_gen)[0]
                model_completion = cur_response
                cur_model_answer = clean_answer(cur_response) if args.dataset_name=='strqa' else clean_answer_gsm8k(cur_response)
                model_answer = cur_model_answer
                is_cor = is_correct(cur_model_answer, all_gt_answers[i]) if args.dataset_name=='strqa' else is_correct_gsm8k(cur_model_answer, all_gt_answers[i])
                input_text = all_input_texts[i]
                correct_rate += is_cor
                print('\n# Correct answers:',correct_rate,'\n')
                result_dict['is_correct'].append(is_cor)
                result_dict['model_answer'].append(model_answer)
                result_dict['model_completion'].append(model_completion)
                result_dict['full_input_text'].append(input_text)
            else:
                response = tokenizer.decode(response[0][0], skip_special_tokens=True) # Note: [0] only needed because of temp fix to loop through num_ret_seq
                for check_gen in checkgens: # Fix generation stopping errors
                    # before_trunc = response
                    response = response.split(check_gen)[0]
                    # if before_trunc=="":
                    #     print(i)
                responses.append({'prompt':prompts[i],
                                    'response1':response})
        else:
            if args.dataset_name=='strqa' or args.dataset_name=='gsm8k':
                is_cor, model_answer, model_completion, input_text = [], [], [], []
                for j in range(args.num_ret_seq):
                    cur_response = tokenizer.decode(response[j][0], skip_special_tokens=True) # Note: [0] only needed because of temp fix to loop through num_ret_seq
                    for check_gen in checkgens: # Fix generation stopping errors
                        cur_response = cur_response.split(check_gen)[0]
                    model_completion.append(cur_response)
                    cur_model_answer = clean_answer(cur_response) if args.dataset_name=='strqa' else clean_answer_gsm8k(cur_response)
                    model_answer.append(cur_model_answer)
                    is_cor.append(is_correct(cur_model_answer, all_gt_answers[i]) if args.dataset_name=='strqa' else is_correct_gsm8k(cur_model_answer, all_gt_answers[i]))
                    input_text.append(all_input_texts[i])
                correct_rate += sum(is_cor)
                print('\n# Correct answers:',correct_rate,'\n')
                result_dict['is_correct'].append(is_cor)
                result_dict['model_answer'].append(model_answer)
                result_dict['model_completion'].append(model_completion)
                result_dict['full_input_text'].append(input_text)
            else:
                resp_dict = {'prompt':prompts[i]}
                for j in range(args.num_ret_seq):
                    cur_response = tokenizer.decode(response[j][0], skip_special_tokens=True) # Note: [0] only needed because of temp fix to loop through num_ret_seq
                    for check_gen in checkgens: # Fix generation stopping errors
                        cur_response = cur_response.split(check_gen)[0]
                    resp_dict['response'+str(j+1)] = cur_response
                    # print(i,j,'Response:',cur_response,'\n')
                responses.append(resp_dict)
    
    # batches = [(0,10)]
    # for batch_start,batch_end in batches:
    #     tokenized_prompt = tokenizer(prompts[batch_start:batch_end], return_tensors = 'pt').input_ids
    #     tokenized_prompt = tokenized_prompt.to(device)
    #     response = model.generate(tokenized_prompt, max_new_tokens=512, num_beams=1, do_sample=False, num_return_sequences=1,
    #                                 eos_token_id=period_token_id,
    #                                 bad_words_ids=question_framing_ids + [tokenized_prompt.tolist()[0]]
    #                                 )[:, tokenized_prompt.shape[-1]:]
    #     response = tokenizer.decode(response, skip_special_tokens=True)
    #     for i,resp in enumerate(response):
    #         for check_gen in checkgens: # Fix generation stopping errors
    #             resp = resp.split(check_gen)[0]
    #         responses.append({'prompt':prompts[batch_start+i],
    #                         'response1':resp})
    
    print('Saving model responses..')
    if args.hallu_check_prompt is None:
        gen_type = 'sampled' if args.do_sample else 'greedy'
        save_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{gen_type}_responses_{args.use_split}{args.len_dataset}.json'
    else:
        save_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_hallucheck{args.hallu_check_prompt}_responses_{args.use_split}{args.len_dataset}.json'
    
    # with open(save_fname, 'r') as read_file:
    #     responses = []
    #     for line in read_file:
    #         responses.append(json.loads(line))
    
    # Fix llama-2 generation issue
    # for i,row in enumerate(responses):
    #     for j in range(args.num_ret_seq):
    #         resp = row['response'+str(j+1)]
    #         responses[i]['response'+str(j+1)] = resp.split("\n")[0]
        # if resp.split("\n")[0]!=resp:
        #     print(i,"\n",resp,"\n\n",resp.split("\n")[0])
    
    if args.dataset_name=='strqa' or args.dataset_name=='gsm8k':
        with open(save_fname, 'w') as f:
            json.dump(result_dict, f)
        np.save(f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{gen_type}_responses_{args.use_split}{args.len_dataset}_oom_idxs.npy',oom_err_idxs)
    else:
        with open(save_fname, 'w') as outfile:
            for entry in responses:
                json.dump(entry, outfile)
                outfile.write('\n')

    
    print('Getting labels for model responses..')
    labels = []
    if args.dataset_name=='strqa' or args.dataset_name=='gsm8k':
        pass
    else:
        rouge = evaluate.load('rouge')
        exact_match_metric = evaluate.load("exact_match")
        squad_metrics = evaluate.load('squad')
        for i,batch in tqdm(enumerate(list(dataset.take(args.len_dataset))[start_at:])): # one row at a time
            if args.num_ret_seq==1:
                labels_dict = {'exact_match': 0.0,
                                'rouge1_to_target':0.0,
                                'rouge2_to_target':0.0,
                                'rougeL_to_target':0.0,
                                'squad_f1':0.0}
            else:
                labels_dict = {}
                for j in range(args.num_ret_seq):
                    labels_dict['exact_match_response'+str(j+1)]=0.0
                    labels_dict['rouge1_to_target_response'+str(j+1)]=0.0
                    labels_dict['rouge2_to_target_response'+str(j+1)]=0.0
                    labels_dict['rougeL_to_target_response'+str(j+1)]=0.0
                    labels_dict['squad_f1_response'+str(j+1)]=0.0
            if args.dataset_name=='nq_open':
                reference_answers = batch['answer'] 
            elif args.dataset_name=='trivia_qa':
                reference_answers_unformatted = batch['answer']
                reference_answers = reference_answers_unformatted['aliases'] + reference_answers_unformatted['normalized_aliases']
            elif args.dataset_name=='cnn_dailymail':
                reference_answers = [batch['highlights']]
            for answer in reference_answers:
                for j in range(args.num_ret_seq):
                    resp_wise_label_name = '_response'+str(j+1) if args.num_ret_seq>1 else ''
                    # predictions, predictions_dict = [responses[j]['response1'].lstrip()], [{'prediction_text':responses[j]['response1'].lstrip()}]
                    # references, references_dict = [answer], [{'answers':{'text':[answer]}}]
                    predictions = [responses[i]['response'+str(j+1)].lstrip()]
                    references = [answer]
                    results = exact_match_metric.compute(predictions=predictions,
                                                            references=references,
                                                            ignore_case=True,
                                                            ignore_punctuation=True)
                    labels_dict['exact_match' + resp_wise_label_name] = max(results['exact_match'], labels_dict['exact_match' + resp_wise_label_name])
                    rouge_results = rouge.compute(predictions=predictions, references=references)
                    for rouge_type in ['rouge1','rouge2','rougeL']:
                        labels_dict[rouge_type + '_to_target' + resp_wise_label_name] = max(rouge_results[rouge_type],
                                                                        labels_dict[rouge_type + '_to_target' + resp_wise_label_name])
                    squad_f1 = my_squad_f1_score(predictions[0],references[0])
                    labels_dict['squad_f1' + resp_wise_label_name] = max(squad_f1, labels_dict['squad_f1' + resp_wise_label_name])

            labels.append(labels_dict)


        print('Saving labels..')
        if args.hallu_check_prompt is None:
            save_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_{gen_type}_responses_labels_{args.use_split}{args.len_dataset}.json'
        else:
            save_fname = f'{args.save_path}/responses/{args.model_name}_{args.dataset_name}_hallucheck{args.hallu_check_prompt}_responses_labels_{args.use_split}{args.len_dataset}.json'
        with open(save_fname, 'w') as outfile:
            for entry in labels:
                json.dump(entry, outfile)
                outfile.write('\n')
    

if __name__ == '__main__':
    main()