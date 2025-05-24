# python get_prompt_responses_factual_onlylabels.py hl_llama_7B city_country --len_dataset 0 --start_at 0 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data
# python get_prompt_responses_factual_onlylabels.py hl_llama_7B city_country --len_dataset 0 --start_at 0 --use_split test --device 0 --save_path /home/local/data/ms/honest_llama_data
# accelerate launch --num_processes 2 --multi_gpu  get_prompt_responses_factual_parallel.py hl_llama_7B city_country --trivia_prompt_format True --len_dataset 0 --start_at 0 --use_split test --device 0 --save_path /home/local/data/ms/honest_llama_data
# accelerate launch --num_processes 2 --multi_gpu  get_prompt_responses_factual_parallel_fast.py hl_llama_7B city_country --len_dataset 1000 --start_at 0 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10
# python get_prompt_responses_factual_onlylabels.py hl_llama_7B city_country --len_dataset 0 --start_at 0 --use_split test --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10

# python get_prompt_responses_factual_onlylabels.py hl_llama_7B player_date_birth --len_dataset 0 --start_at 0 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data
# python get_prompt_responses_factual_onlylabels.py hl_llama_7B player_date_birth --len_dataset 0 --start_at 0 --use_split test --device 0 --save_path /home/local/data/ms/honest_llama_data
# accelerate launch --num_processes 2 --main_process_port 29502 get_prompt_responses_factual_parallel_fast.py hl_llama_7B player_date_birth --len_dataset 1000 --start_at 0 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10
# accelerate launch --num_processes 2 get_prompt_responses_factual_parallel_fast.py hl_llama_7B player_date_birth --len_dataset 0 --start_at 0 --use_split test --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10

# python get_prompt_responses_factual_onlylabels.py hl_llama_7B movie_cast --len_dataset 0 --start_at 0 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data
# python get_prompt_responses_factual_onlylabels.py hl_llama_7B movie_cast --len_dataset 0 --start_at 0 --use_split test --device 0 --save_path /home/local/data/ms/honest_llama_data
# accelerate launch --num_processes 2 --multi_gpu get_prompt_responses_factual_parallel_fast.py hl_llama_7B movie_cast --len_dataset 1000 --start_at 0 --use_split train --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10
# accelerate launch --num_processes 2 --multi_gpu get_prompt_responses_factual_parallel_fast.py hl_llama_7B movie_cast --len_dataset 0 --start_at 0 --use_split test --device 0 --save_path /home/local/data/ms/honest_llama_data --do_sample True --num_ret_seq 10


##### Gemma 2B ########
accelerate launch --num_processes 2 --multi_gpu  get_prompt_responses_factual_parallel.py gemma_2B city_country --len_dataset 0 --start_at 0 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data;
printf "City train completed";
accelerate launch --num_processes 2 --multi_gpu  get_prompt_responses_factual_parallel.py gemma_2B city_country --len_dataset 0 --start_at 0 --use_split test --device 0 --save_path /home/local/data/ms/honest_llama_data;
printf "City test completed";

accelerate launch --num_processes 2 --multi_gpu  get_prompt_responses_factual_parallel.py gemma_2B player_date_birth --len_dataset 0 --start_at 0 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data;
printf "Player train completed";
accelerate launch --num_processes 2 --multi_gpu  get_prompt_responses_factual_parallel.py gemma_2B player_date_birth --len_dataset 0 --start_at 0 --use_split test --device 0 --save_path /home/local/data/ms/honest_llama_data;
printf "Player test completed";

accelerate launch --num_processes 2 --multi_gpu  get_prompt_responses_factual_parallel.py gemma_2B movie_cast --len_dataset 0 --start_at 0 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data;
printf "Movie train completed";
accelerate launch --num_processes 2 --multi_gpu  get_prompt_responses_factual_parallel.py gemma_2B movie_cast --len_dataset 0 --start_at 0 --use_split test --device 0 --save_path /home/local/data/ms/honest_llama_data;
printf "Movie test completed";