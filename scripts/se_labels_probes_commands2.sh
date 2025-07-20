# python get_uncertainty_scores.py llama3.1_8B_Instruct trivia_qa --file_name sampledplus_responses_train5000 --save_path /home/local/data/ms/honest_llama_data --num_samples 10;
# python get_uncertainty_scores.py llama3.1_8B_Instruct city_country --file_name sampled_responses_train1000 --save_path /home/local/data/ms/honest_llama_data --num_samples 10;
# python get_uncertainty_scores.py llama3.1_8B_Instruct player_date_birth --file_name sampled_responses_train1000 --save_path /home/local/data/ms/honest_llama_data --num_samples 10;
# python get_uncertainty_scores.py llama3.1_8B_Instruct movie_cast --file_name sampled_responses_train1000 --save_path /home/local/data/ms/honest_llama_data --num_samples 10;

python get_semantic_entropy.py llama3.1_8B_Instruct trivia_qa --file_name sampledplus_responses_train5000 --len_dataset 1000 --save_path /home/local/data/ms/honest_llama_data --num_samples 10;
python get_semantic_entropy.py llama3.1_8B_Instruct city_country --file_name sampled_responses_train1000 --save_path /home/local/data/ms/honest_llama_data --num_samples 10;
python get_semantic_entropy.py llama3.1_8B_Instruct player_date_birth --file_name sampled_responses_train1000 --save_path /home/local/data/ms/honest_llama_data --num_samples 10;
python get_semantic_entropy.py llama3.1_8B_Instruct movie_cast --file_name sampled_responses_train1000 --save_path /home/local/data/ms/honest_llama_data --num_samples 10;