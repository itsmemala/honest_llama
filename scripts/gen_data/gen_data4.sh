# python get_prompt_responses_factual.py gemma_2B gsm8k --len_dataset 5000 --start_at 0 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data
# python get_prompt_responses_factual.py gemma_2B gsm8k --len_dataset 1800 --start_at 0 --use_split test --device 0 --save_path /home/local/data/ms/honest_llama_data
# python get_prompt_responses_factual.py gemma_2B gsm8k --len_dataset 1000 --start_at 0 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10
# python get_prompt_responses_factual.py gemma_2B gsm8k --len_dataset 2000 --start_at 1000 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10
# python get_prompt_responses_factual.py gemma_2B gsm8k --len_dataset 3000 --start_at 2000 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10
# python get_prompt_responses_factual.py gemma_2B gsm8k --len_dataset 4000 --start_at 3000 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10
# python get_prompt_responses_factual.py gemma_2B gsm8k --len_dataset 5000 --start_at 4000 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10
# python get_prompt_responses_factual.py gemma_2B gsm8k --len_dataset 1800 --start_at 0 --use_split test --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10

python combine_response_files_gsm.py --model gemma_2B --save_path /home/local/data/ms/honest_llama_data

python get_activations.py gemma_2B gsm8k --token answer_last --file_name gsm8k_greedy_responses_train5000  --device 0 --save_path /home/local/data/ms/honest_llama_data
python get_activations.py gemma_2B gsm8k --token answer_last --file_name gsm8k_greedy_responses_test1800  --device 0 --save_path /home/local/data/ms/honest_llama_data
python get_activations.py gemma_2B gsm8k --token answer_last --file_name gsm8k_sampledplus_responses_train2000  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 11
python get_activations.py gemma_2B gsm8k --token answer_last --file_name gsm8k_sampled_responses_test1800  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 10