# python get_prompt_responses_factual.py llama3.1_8B_Instruct strqa --len_dataset 5000 --start_at 0 --use_split train --device 0 --save_path ~/Desktop/honest_llama_data
# python get_prompt_responses_factual.py llama3.1_8B_Instruct strqa --len_dataset 5000 --start_at 0 --use_split train --device 0 --save_path ~/Desktop/honest_llama_data --do_sample True --num_ret_seq 8

python get_activations.py llama3.1_8B_Instruct strqa --token answer_last --file_name strqa_baseline_responses_train  --device 0 --save_path /home/local/data/ms/honest_llama_data
python get_activations.py llama3.1_8B_Instruct strqa --token answer_last --file_name strqa_baseline_responses_test  --device 0 --save_path /home/local/data/ms/honest_llama_data
python get_activations.py llama3.1_8B_Instruct strqa --token answer_last --file_name strqa_sampledplus_responses_train  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 9
python get_activations.py llama3.1_8B_Instruct strqa --token answer_last --file_name strqa_sampled_responses_test  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 8