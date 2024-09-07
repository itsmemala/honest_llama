# 3k start at 1.8k #
python get_prompt_responses_factual.py hl_llama_7B nq_open --len_dataset 5000 --start_at 2000 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data
python get_prompt_responses_factual.py hl_llama_7B gsm8k --len_dataset 5000 --start_at 2000 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data
python get_prompt_responses_factual.py alpaca_7B gsm8k --len_dataset 5000 --start_at 2000 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data