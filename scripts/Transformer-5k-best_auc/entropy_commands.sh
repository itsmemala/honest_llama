python get_uncertainty_scores.py hl_llama_7B trivia_qa --file_name greedy_responses_train5000 --save_path /home/local/data/ms/honest_llama_data
python get_uncertainty_scores.py hl_llama_7B trivia_qa --file_name greedy_responses_validation1800 --save_path /home/local/data/ms/honest_llama_data
python get_baselines.py hl_llama_7B trivia_qa --len_dataset 5000 --train_labels_file_name greedy_responses_labels_train5000 --test_labels_file_name greedy_responses_labels_validation1800 --train_uncertainty_values_file_name greedy_responses_train5000 --test_uncertainty_values_file_name greedy_responses_validation1800 --save_path /home/local/data/ms/honest_llama_data

python get_uncertainty_scores.py hl_llama_7B nq_open --file_name greedy_responses_train5000 --save_path /home/local/data/ms/honest_llama_data
python get_uncertainty_scores.py hl_llama_7B nq_open --file_name greedy_responses_validation1800 --save_path /home/local/data/ms/honest_llama_data
python get_baselines.py hl_llama_7B nq_open --len_dataset 5000 --train_labels_file_name greedy_responses_labels_train5000 --test_labels_file_name greedy_responses_labels_validation1800 --train_uncertainty_values_file_name greedy_responses_train5000 --test_uncertainty_values_file_name greedy_responses_validation1800 --save_path /home/local/data/ms/honest_llama_data

python get_uncertainty_scores.py hl_llama_7B strqa --file_name baseline_responses_train --save_path /home/local/data/ms/honest_llama_data
python get_uncertainty_scores.py hl_llama_7B strqa --file_name baseline_responses_test --save_path /home/local/data/ms/honest_llama_data
python get_baselines.py hl_llama_7B strqa --len_dataset 1832 --train_labels_file_name baseline_responses_train --test_labels_file_name baseline_responses_test --train_uncertainty_values_file_name baseline_responses_train --test_uncertainty_values_file_name baseline_responses_test --save_path /home/local/data/ms/honest_llama_data

python get_uncertainty_scores.py alpaca_7B trivia_qa --file_name greedy_responses_train5000 --save_path /home/local/data/ms/honest_llama_data
python get_uncertainty_scores.py alpaca_7B trivia_qa --file_name greedy_responses_validation1800 --save_path /home/local/data/ms/honest_llama_data

python get_uncertainty_scores.py alpaca_7B nq_open --file_name greedy_responses_train5000 --save_path /home/local/data/ms/honest_llama_data
python get_uncertainty_scores.py alpaca_7B nq_open --file_name greedy_responses_validation1800 --save_path /home/local/data/ms/honest_llama_data
python get_baselines.py alpaca_7B nq_open --len_dataset 5000 --train_labels_file_name greedy_responses_labels_train5000 --test_labels_file_name greedy_responses_labels_validation1800 --train_uncertainty_values_file_name greedy_responses_train5000 --test_uncertainty_values_file_name greedy_responses_validation1800 --save_path /home/local/data/ms/honest_llama_data

python get_uncertainty_scores.py alpaca_7B strqa --file_name baseline_responses_train --save_path /home/local/data/ms/honest_llama_data
python get_uncertainty_scores.py alpaca_7B strqa --file_name baseline_responses_test --save_path /home/local/data/ms/honest_llama_data
python get_baselines.py alpaca_7B strqa --len_dataset 1832 --train_labels_file_name baseline_responses_train --test_labels_file_name baseline_responses_test --train_uncertainty_values_file_name baseline_responses_train --test_uncertainty_values_file_name baseline_responses_test --save_path /home/local/data/ms/honest_llama_data