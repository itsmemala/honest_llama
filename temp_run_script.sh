python get_semantic_entropy.py alpaca_7B trivia_qa --num_samples 11 --file_name sampledplus_responses_train2000 --save_path ~/Desktop/honest_llama_data

# python get_uncertainty_scores.py alpaca_7B nq_open --file_name sampledplus_responses_train2000 --save_path ~/Desktop/honest_llama_data --num_samples 11
# python get_semantic_entropy.py alpaca_7B nq_open --num_samples 11 --file_name sampledplus_responses_train2000 --save_path ~/Desktop/honest_llama_data

python get_prompt_responses_factual.py hl_llama_7B nq_open --len_dataset 2000 --start_at 0 --use_split train --save_path ~/Desktop/honest_llama_data