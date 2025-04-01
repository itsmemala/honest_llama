# python get_prompt_responses_factual.py alpaca_7B trivia_qa --len_dataset 1800 --start_at 0 --use_split validation --device 0 --save_path ~/Desktop/honest_llama_data --do_sample True --num_ret_seq 10
# python get_prompt_responses_factual.py vicuna_7B trivia_qa --len_dataset 1800 --start_at 0 --use_split validation --device 0 --save_path ~/Desktop/honest_llama_data --do_sample True --num_ret_seq 10
# python get_prompt_responses_factual.py alpaca_7B nq_open --len_dataset 1800 --start_at 0 --use_split validation --device 0 --save_path ~/Desktop/honest_llama_data --do_sample True --num_ret_seq 10
# python get_prompt_responses_factual.py vicuna_7B nq_open --len_dataset 1800 --start_at 0 --use_split validation --device 0 --save_path ~/Desktop/honest_llama_data --do_sample True --num_ret_seq 10

# python get_activations.py hl_llama_7B nq_open --token answer_last --file_name nq_open_sampled_responses_validation1800  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 10
# python get_activations.py alpaca_7B nq_open --token answer_last --file_name nq_open_sampled_responses_validation1800  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 10
python get_activations.py vicuna_7B nq_open --token answer_last --file_name nq_open_sampled_responses_validation1800  --device 0 --save_path /home/local/data/ms/honest_llama --num_samples 10
# python get_activations.py alpaca_7B trivia_qa --token answer_last --file_name trivia_qa_sampled_responses_validation1800  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 10
# python get_activations.py vicuna_7B trivia_qa --token answer_last --file_name trivia_qa_sampled_responses_validation1800  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 10