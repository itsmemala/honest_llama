############## Transformer 1 layer - n ##################################
# python analyse_crossdataset_bce_transfmr.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name T42_hl_llama_7B_trivia_qa_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path /home/local/data/ms/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce_transfmr.py hl_llama_7B nq_open --using_act layer --token answer_last --probes_file_name T42_hl_llama_7B_nq_open_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path /home/local/data/ms/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce_transfmr.py hl_llama_7B strqa --using_act layer --token answer_last --probes_file_name T42_hl_llama_7B_strqa_baseline_responses_train_1832_1_layerFalse_answer_last_transformer_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path /home/local/data/ms/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce_transfmr.py alpaca_7B trivia_qa --using_act layer --token answer_last --probes_file_name T42_alpaca_7B_trivia_qa_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path /home/local/data/ms/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce_transfmr.py alpaca_7B nq_open --using_act layer --token answer_last --probes_file_name T42_alpaca_7B_nq_open_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path /home/local/data/ms/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce_transfmr.py alpaca_7B strqa --using_act layer --token answer_last --probes_file_name T42_alpaca_7B_strqa_baseline_responses_train_1832_1_layerFalse_answer_last_transformer_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path /home/local/data/ms/honest_llama_data --seed_list 42,101,2650

# ############## Transformer 2 layer - n ########################################
# python analyse_crossdataset_bce_transfmr.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name T42_hl_llama_7B_trivia_qa_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer2_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path /home/local/data/ms/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce_transfmr.py hl_llama_7B nq_open --using_act layer --token answer_last --probes_file_name T42_hl_llama_7B_nq_open_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer2_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path /home/local/data/ms/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce_transfmr.py hl_llama_7B strqa --using_act layer --token answer_last --probes_file_name T42_hl_llama_7B_strqa_baseline_responses_train_1832_1_layerFalse_answer_last_transformer2_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path /home/local/data/ms/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce_transfmr.py alpaca_7B trivia_qa --using_act layer --token answer_last --probes_file_name T42_alpaca_7B_trivia_qa_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer2_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path /home/local/data/ms/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce_transfmr.py alpaca_7B nq_open --using_act layer --token answer_last --probes_file_name T42_alpaca_7B_nq_open_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer2_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path /home/local/data/ms/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce_transfmr.py alpaca_7B strqa --using_act layer --token answer_last --probes_file_name T42_alpaca_7B_strqa_baseline_responses_train_1832_1_layerFalse_answer_last_transformer2_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path /home/local/data/ms/honest_llama_data --seed_list 42,101,2650

################################### transformer n*aug 1 layer ####################################
python analyse_crossdataset_bce_transfmr.py hl_llama_7B strqa --using_act layer --token answer_last --probes_file_name T42_hl_llama_7B_strqa_baseline_responses_train_16479_1_layerFalse_answer_last_transformer_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

################################# transformer n*aug 1 layer : supcon ##########################
python analyse_crossdataset_bce_transfmr.py hl_llama_7B strqa --using_act layer --token answer_last --probes_file_name T42_hl_llama_7B_strqa_baseline_responses_train_16479_1_layerFalse_answer_last_transformer_supconv2_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

################################# transformer n*aug 1 layer : supcon+ ##########################
python analyse_crossdataset_bce_transfmr.py hl_llama_7B strqa --using_act layer --token answer_last --probes_file_name T42_hl_llama_7B_strqa_baseline_responses_train_16479_1_layerFalse_answer_last_transformer_supconv2_pos_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

################################# transformer n*aug 1 layer : supcon* ##########################
python analyse_crossdataset_bce_transfmr.py hl_llama_7B strqa --using_act layer --token answer_last --probes_file_name T42_hl_llama_7B_strqa_baseline_responses_train_16479_1_layerFalse_answer_last_transformer_supconv2_wp_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

################################# transformer n*aug 1 layer : supcon+* ##########################
python analyse_crossdataset_bce_transfmr.py hl_llama_7B strqa --using_act layer --token answer_last --probes_file_name T42_hl_llama_7B_strqa_baseline_responses_train_16479_1_layerFalse_answer_last_transformer_supconv2_pos_wp_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650