################# Linear ###############
# python analyse_crossdataset_bce.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC42_hl_llama_7B_ood_trivia_qa_nq_open_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path /home/local/data/ms/honest_llama_data
# python analyse_crossdataset_bce.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC101_hl_llama_7B_ood_trivia_qa_nq_open_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path /home/local/data/ms/honest_llama_data
# python analyse_crossdataset_bce.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC2650_hl_llama_7B_ood_trivia_qa_nq_open_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path /home/local/data/ms/honest_llama_data

# python analyse_crossdataset_bce.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC42_hl_llama_7B_ood_trivia_qa_strqa_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path /home/local/data/ms/honest_llama_data
# python analyse_crossdataset_bce.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC101_hl_llama_7B_ood_trivia_qa_strqa_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path /home/local/data/ms/honest_llama_data
# python analyse_crossdataset_bce.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC2650_hl_llama_7B_ood_trivia_qa_strqa_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path /home/local/data/ms/honest_llama_data

# ###################### Non-linear ##############

# python analyse_crossdataset_bce.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC42_hl_llama_7B_ood_trivia_qa_nq_open_5000_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path /home/local/data/ms/honest_llama_data
# python analyse_crossdataset_bce.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC101_hl_llama_7B_ood_trivia_qa_nq_open_5000_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path /home/local/data/ms/honest_llama_data
# python analyse_crossdataset_bce.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC2650_hl_llama_7B_ood_trivia_qa_nq_open_5000_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path /home/local/data/ms/honest_llama_data

# python analyse_crossdataset_bce.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC42_hl_llama_7B_ood_trivia_qa_strqa_5000_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path /home/local/data/ms/honest_llama_data
# python analyse_crossdataset_bce.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC101_hl_llama_7B_ood_trivia_qa_strqa_5000_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path /home/local/data/ms/honest_llama_data
# python analyse_crossdataset_bce.py hl_llama_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC2650_hl_llama_7B_ood_trivia_qa_strqa_5000_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path /home/local/data/ms/honest_llama_data

################# Linear ###############
# python analyse_crossdataset_bce.py hl_llama_7B nq_open --using_act layer --token answer_last --probes_file_name NLSC42_hl_llama_7B_ood_nq_open_trivia_qa_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce.py hl_llama_7B nq_open --using_act layer --token answer_last --probes_file_name NLSC42_hl_llama_7B_ood_nq_open_strqa_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

# ###################### Non-linear ##############

# python analyse_crossdataset_bce.py hl_llama_7B nq_open --using_act layer --token answer_last --probes_file_name NLSC42_hl_llama_7B_ood_nq_open_trivia_qa_5000_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce.py hl_llama_7B nq_open --using_act layer --token answer_last --probes_file_name NLSC42_hl_llama_7B_ood_nq_open_strqa_5000_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

# ################# Linear ###############
# python analyse_crossdataset_bce.py hl_llama_7B strqa --using_act layer --token answer_last --probes_file_name NLSC42_hl_llama_7B_ood_strqa_trivia_qa_1832_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce.py hl_llama_7B strqa --using_act layer --token answer_last --probes_file_name NLSC42_hl_llama_7B_ood_strqa_nq_open_1832_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

# ###################### Non-linear ##############

# python analyse_crossdataset_bce.py hl_llama_7B strqa --using_act layer --token answer_last --probes_file_name NLSC42_hl_llama_7B_ood_strqa_trivia_qa_1832_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce.py hl_llama_7B strqa --using_act layer --token answer_last --probes_file_name NLSC42_hl_llama_7B_ood_strqa_nq_open_1832_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

################# Linear ###############
# python analyse_crossdataset_bce.py alpaca_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC42_alpaca_7B_ood_trivia_qa_nq_open_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce.py alpaca_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC42_alpaca_7B_ood_trivia_qa_strqa_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

###################### Non-linear ##############

python analyse_crossdataset_bce.py alpaca_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC42_alpaca_7B_ood_trivia_qa_nq_open_5000_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

python analyse_crossdataset_bce.py alpaca_7B trivia_qa --using_act layer --token answer_last --probes_file_name NLSC42_alpaca_7B_ood_trivia_qa_strqa_5000_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

################# Linear ###############
# python analyse_crossdataset_bce.py alpaca_7B nq_open --using_act layer --token answer_last --probes_file_name NLSC42_alpaca_7B_ood_nq_open_trivia_qa_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce.py alpaca_7B nq_open --using_act layer --token answer_last --probes_file_name NLSC42_alpaca_7B_ood_nq_open_strqa_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

###################### Non-linear ##############

python analyse_crossdataset_bce.py alpaca_7B nq_open --using_act layer --token answer_last --probes_file_name NLSC42_alpaca_7B_ood_nq_open_trivia_qa_5000_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

python analyse_crossdataset_bce.py alpaca_7B nq_open --using_act layer --token answer_last --probes_file_name NLSC42_alpaca_7B_ood_nq_open_strqa_5000_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

################# Linear ###############
# python analyse_crossdataset_bce.py alpaca_7B strqa --using_act layer --token answer_last --probes_file_name NLSC42_alpaca_7B_ood_strqa_trivia_qa_1832_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

# python analyse_crossdataset_bce.py alpaca_7B strqa --using_act layer --token answer_last --probes_file_name NLSC42_alpaca_7B_ood_strqa_nq_open_1832_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

###################### Non-linear ##############

python analyse_crossdataset_bce.py alpaca_7B strqa --using_act layer --token answer_last --probes_file_name NLSC42_alpaca_7B_ood_strqa_trivia_qa_1832_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650

python analyse_crossdataset_bce.py alpaca_7B strqa --using_act layer --token answer_last --probes_file_name NLSC42_alpaca_7B_ood_strqa_nq_open_1832_1_layerFalse_answer_last_individual_non_linear_4_hallu_pos_bs128_epochs50_ --probes_file_name_concat ba --lr_list 0.00005,0.0005,0.005,0.05,0.5 --best_threshold True --save_path ~/Desktop/honest_llama_data --seed_list 42,101,2650