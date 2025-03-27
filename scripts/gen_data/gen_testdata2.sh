# python get_prompt_responses_factual.py hl_llama_7B strqa --len_dataset 5000 --start_at 0 --use_split train --device 0 --save_path ~/Desktop/honest_llama_data --do_sample True --num_ret_seq 8
# python get_prompt_responses_factual.py hl_llama_7B gsm8k --len_dataset 1800 --start_at 0 --use_split test --device 0 --save_path ~/Desktop/honest_llama_data --do_sample True --num_ret_seq 10
# python get_prompt_responses_factual.py alpaca_7B gsm8k --len_dataset 1800 --start_at 0 --use_split test --device 0 --save_path ~/Desktop/honest_llama_data --do_sample True --num_ret_seq 10
# python get_prompt_responses_factual.py vicuna_7B gsm8k --len_dataset 1800 --start_at 0 --use_split test --device 0 --save_path ~/Desktop/honest_llama_data --do_sample True --num_ret_seq 10

python get_activations.py hl_llama_7B gsm8k --token answer_last --file_name gsm8k_sampled_responses_test1800  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 10
python get_activations.py alpaca_7B gsm8k --token answer_last --file_name gsm8k_sampled_responses_test1800  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 10
python get_activations.py vicuna_7B gsm8k --token answer_last --file_name gsm8k_sampled_responses_test1800  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 10