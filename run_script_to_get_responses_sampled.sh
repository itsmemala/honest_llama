# 2k #
# accelerate launch --num_processes 2 get_prompt_responses_factual_parallel.py alpaca_7B nq_open --len_dataset 2000 --start_at 0 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10

# 3k (till 5k start at 2k) #
# accelerate launch --num_processes 2 get_prompt_responses_factual_parallel.py hl_llama_7B trivia_qa --len_dataset 5000 --start_at 2000 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10
# accelerate launch --num_processes 2 get_prompt_responses_factual_parallel.py hl_llama_7B nq_open --len_dataset 5000 --start_at 2000 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10
# accelerate launch --num_processes 2 get_prompt_responses_factual_parallel.py hl_llama_7B gsm8k --len_dataset 5000 --start_at 2000 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10
# accelerate launch --num_processes 2 get_prompt_responses_factual_parallel.py alpaca_7B trivia_qa --len_dataset 5000 --start_at 2000 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10
accelerate launch --num_processes 2 get_prompt_responses_factual_parallel.py alpaca_7B nq_open --len_dataset 5000 --start_at 2000 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10
accelerate launch --num_processes 2 get_prompt_responses_factual_parallel.py alpaca_7B gsm8k --len_dataset 5000 --start_at 2000 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10