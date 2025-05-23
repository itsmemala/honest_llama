# python get_prompt_responses_factual_onlylabels.py llama3.1_8B_Instruct nq_open --len_dataset 5000 --start_at 0 --use_split train --device 0 --save_path ~/Desktop/honest_llama_data
# # python get_prompt_responses_factual_onlylabels.py llama3.1_8B_Instruct nq_open --len_dataset 1800 --start_at 0 --use_split validation --device 0 --save_path ~/Desktop/honest_llama_data
# python get_prompt_responses_factual_onlylabels.py llama3.1_8B_Instruct nq_open --len_dataset 5000 --start_at 0 --use_split train --device 0 --save_path ~/Desktop/honest_llama_data --do_sample True --num_ret_seq 10
# python get_prompt_responses_factual_onlylabels.py llama3.1_8B_Instruct nq_open --len_dataset 1800 --start_at 0 --use_split validation --device 0 --save_path ~/Desktop/honest_llama_data --do_sample True --num_ret_seq 10

# python combine_response_files_nq.py --model llama3.1_8B_Instruct --save_path /home/local/data/ms/honest_llama_data

python get_activations.py llama3.1_8B_Instruct nq_open --token answer_last --file_name nq_open_greedy_responses_train5000  --device 0 --save_path /home/local/data/ms/honest_llama_data
python get_activations.py llama3.1_8B_Instruct nq_open --token answer_last --file_name nq_open_greedy_responses_validation1800  --device 0 --save_path /home/local/data/ms/honest_llama_data
python get_activations.py llama3.1_8B_Instruct nq_open --token answer_last --file_name nq_open_sampledplus_responses_train5000  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 11
python get_activations.py llama3.1_8B_Instruct nq_open --token answer_last --file_name nq_open_sampled_responses_validation1800  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 10