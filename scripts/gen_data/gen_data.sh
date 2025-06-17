# python get_prompt_responses_factual.py gemma_7B trivia_qa --len_dataset 5000 --start_at 0 --use_split train --device 0 --save_path /home/local/data/ms/honest_llama_data
# printf "Greedy train completed";
# python get_prompt_responses_factual.py gemma_7B trivia_qa --len_dataset 1800 --start_at 0 --use_split validation --device 0 --save_path /home/local/data/ms/honest_llama_data
# printf "Greedy test completed";
# python get_prompt_responses_factual.py gemma_7B trivia_qa --len_dataset 5000 --start_at 0 --use_split train --device 0 --save_path ~/Desktop/honest_llama_data --do_sample True --num_ret_seq 10
# printf "Sampled train completed";
# python get_prompt_responses_factual.py gemma_7B trivia_qa --len_dataset 1800 --start_at 0 --use_split validation --device 0 --save_path ~/Desktop/honest_llama_data --do_sample True --num_ret_seq 10
# printf "Sampled test completed";

# python combine_response_files_trivia.py --model gemma_7B --save_path /home/local/data/ms/honest_llama_data

# mkdir /home/local/data/ms/honest_llama_data/features/gemma_7B_trivia_qa_answer_last/

# python get_activations.py gemma_7B trivia_qa --token answer_last --file_name trivia_qa_greedy_responses_train5000  --device 0 --save_path /home/local/data/ms/honest_llama_data # Finished
# python get_activations.py gemma_7B trivia_qa --token answer_last --file_name trivia_qa_greedy_responses_validation1800  --device 0 --save_path /home/local/data/ms/honest_llama_data # Finished
# python get_activations.py gemma_7B trivia_qa --token answer_last --file_name trivia_qa_sampledplus_responses_train5000  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 11 # Finished
# python get_activations.py gemma_7B trivia_qa --token answer_last --file_name trivia_qa_sampled_responses_validation1800  --device 0 --save_path /home/local/data/ms/honest_llama_data --num_samples 10 # Finished

# python get_activations.py hl_llama_7B trivia_qa --token answer_last --file_name trivia_qa_dola16to32_responses_test  --device 0 --save_path /home/local/data/ms/honest_llama_data