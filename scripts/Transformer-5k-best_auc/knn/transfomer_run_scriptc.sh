################################# transformer n 1 layer ##########################
# python get_activations.py hl_llama_7B strqa --token answer_last --file_name strqa_baseline_responses_train  --device 0 --save_path /home/local/data/ms/honest_llama_data
# python get_activations.py hl_llama_7B strqa --token answer_last --file_name strqa_baseline_responses_test  --device 0 --save_path /home/local/data/ms/honest_llama_data

python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_knn_hallu_pos --bs 128 --epochs 50 --lr_list 0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer-n --tag main-transformer-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True

################################### transformer n*aug 1 layer ####################################

# python get_activations.py hl_llama_7B strqa --token answer_last --file_name strqa_sampledplus_responses_train  --device 0 --save_path /home/local/data/ms/honest_llama_data
# python get_activations.py hl_llama_7B strqa --token answer_last --file_name strqa_baseline_responses_test  --device 0 --save_path /home/local/data/ms/honest_llama_data

python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_sampledplus_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 16479 --num_folds 1 --using_act layer --token answer_last --method transformer_knn_hallu_pos --bs 128 --epochs 50 --lr_list 0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer-n*aug --tag main-transformer-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True

################################ transformer n 2 layer ##########################################
python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer2_knn_hallu_pos --bs 128 --epochs 50 --lr_list 0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer2-n --tag main-transformer-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True

################################# transformer n 1 layer : ntx ##########################
python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supcon_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_ntx-n --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True --seed_list 42,101,2650 --best_using_auc True

################################# transformer n 1 layer : supcon ##########################
python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n-k20 --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True --top_k 20)
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n-k50 --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True --top_k 50)
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n-c --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True --dist_metric cosine)
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n-m --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True --dist_metric mahalanobis)
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n-mk20 --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True --top_k 20 --dist_metric mahalanobis)
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n-mk50 --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True --top_k 50 --dist_metric mahalanobis)
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n-mw --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True --dist_metric mahalanobis_wgtd)
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n-mwk20 --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True --top_k 20 --dist_metric mahalanobis_wgtd)
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n-mwk50 --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True --top_k 50 --dist_metric mahalanobis_wgtd)
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n-mm --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True --dist_metric mahalanobis_maj)
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n-mmk20 --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True --top_k 20 --dist_metric mahalanobis_maj)
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n-mmk50 --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True --top_k 50 --dist_metric mahalanobis_maj)

################################# transformer n*aug 1 layer : ntx ##########################
python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_sampledplus_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 16479 --num_folds 1 --using_act layer --token answer_last --method transformer_supcon_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_ntx-n*aug --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True   --seed_list 42,101,2650 --best_using_auc True

################################# transformer n*aug 1 layer : supcon ##########################
python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_sampledplus_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 16479 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n*aug --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True   --seed_list 42,101,2650 --best_using_auc True
python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_sampledplus_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 16479 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_knn_hallu_pos --bs 270 --epochs 500 --lr_list 0.000005,0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2-n*aug-b --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True   --seed_list 42,101,2650 --best_using_auc True  --no_batch_sampling True

################################# transformer n 1 layer : supcon+ ##########################
python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_baseline_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 1832 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_pos_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2_pos-n --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True  --seed_list 42,101,2650 --best_using_auc True

################################# transformer n*aug 1 layer : supcon+ ##########################
python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_sampledplus_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 16479 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_pos_knn_hallu_pos --bs 256 --epochs 500 --lr_list 0.000005,0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2_pos-n*aug --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True   --seed_list 42,101,2650 --best_using_auc True
python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_sampledplus_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 16479 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_pos_knn_hallu_pos --bs 270 --epochs 500 --lr_list 0.000005,0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2_pos-n*aug-b --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True   --seed_list 42,101,2650 --best_using_auc True --no_batch_sampling True

################################# transformer n*aug 1 layer : supcon* ##########################
python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_sampledplus_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 16479 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_wp_knn_hallu_pos --bs 270 --epochs 500 --lr_list 0.000005,0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2_wp-n*aug-b --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True   --seed_list 42,101,2650 --best_using_auc True  --no_batch_sampling True

################################# transformer n*aug 1 layer : supcon+* ##########################
python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_sampledplus_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 16479 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_pos_wp_knn_hallu_pos --bs 270 --epochs 500 --lr_list 0.000005,0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2_pos_wp-n*aug-b --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True   --seed_list 42,101,2650 --best_using_auc True  --no_batch_sampling True
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_sampledplus_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 16479 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_pos_wp_kmeans_hallu_pos --bs 270 --epochs 500 --lr_list 0.000005,0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2_pos_wp-n*aug-b --tag main-transformer-contrast-5kbestauc-kmeans --use_best_val_t True   --seed_list 42,101,2650 --best_using_auc True  --no_batch_sampling True --dist_metric mahalanobis_wgtd_centers --top_k 5)
(python get_activations_and_probe_transformer.py hl_llama_7B strqa --train_file_name strqa_sampledplus_responses_train --test_file_name strqa_baseline_responses_test --len_dataset 16479 --num_folds 1 --using_act layer --token answer_last --method transformer_supconv2_pos_wp_knn_hallu_pos --bs 270 --epochs 500 --lr_list 0.000005,0.00005,0.0005,0.005 --save_probes True --save_path /home/local/data/ms/honest_llama_data --fast_mode True  --plot_name transformer_supconv2_pos_wp1-n*aug-b --tag main-transformer-contrast-5kbestauc-knn --use_best_val_t True   --seed_list 42,101,2650 --best_using_auc True  --no_batch_sampling True --sc2_wgt 2 # 1,2)