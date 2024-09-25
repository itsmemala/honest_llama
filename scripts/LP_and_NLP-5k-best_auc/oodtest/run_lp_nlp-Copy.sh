# python get_activations.py hl_llama_7B trivia_qa --token answer_last --file_name trivia_qa_greedy_responses_train5000  --device 0 --save_path ~/Desktop/honest_llama_data
# python get_activations.py hl_llama_7B trivia_qa --token answer_last --file_name trivia_qa_greedy_responses_validation1800  --device 0 --save_path ~/Desktop/honest_llama_data

python get_activations.py hl_llama_7B nq_open --token answer_last --file_name nq_open_greedy_responses_train5000  --device 0 --save_path ~/Desktop/honest_llama_data
# python get_activations.py hl_llama_7B nq_open --token answer_last --file_name nq_open_greedy_responses_validation1800  --device 0 --save_path ~/Desktop/honest_llama_data

# python get_activations.py hl_llama_7B strqa --token answer_last --file_name strqa_baseline_responses_train  --device 0 --save_path ~/Desktop/honest_llama_data
# python get_activations.py hl_llama_7B strqa --token answer_last --file_name strqa_baseline_responses_test  --device 0 --save_path ~/Desktop/honest_llama_data

# python get_activations.py alpaca_7B trivia_qa --token answer_last --file_name trivia_qa_greedy_responses_train5000  --device 0 --save_path ~/Desktop/honest_llama_data
# python get_activations.py alpaca_7B trivia_qa --token answer_last --file_name trivia_qa_greedy_responses_validation1800  --device 0 --save_path ~/Desktop/honest_llama_data

# python get_activations.py alpaca_7B nq_open --token answer_last --file_name nq_open_greedy_responses_train5000  --device 0 --save_path ~/Desktop/honest_llama_data
# python get_activations.py alpaca_7B nq_open --token answer_last --file_name nq_open_greedy_responses_validation1800  --device 0 --save_path ~/Desktop/honest_llama_data

python get_activations.py alpaca_7B strqa --token answer_last --file_name strqa_baseline_responses_train  --device 0 --save_path ~/Desktop/honest_llama_data
python get_activations.py alpaca_7B strqa --token answer_last --file_name strqa_baseline_responses_test  --device 0 --save_path ~/Desktop/honest_llama_data