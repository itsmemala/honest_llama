from matplotlib import pyplot as plt
import numpy as np

save_path = '/home/local/data/ms/honest_llama_data'

fig, axs = plt.subplots(1,1)
# paths = ['NLSC42_hl_llama_7B_trivia_qa_greedy_responses_train5000_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_0.0005_False_fpr_at_recall',
# 'T42_hl_llama_7B_trivia_qa_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer_hallu_pos_bs128_epochs50_5e-05_Falseba_fpr_at_recall',
# 'T42_hl_llama_7B_trivia_qa_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer_kmeans_hallu_pos_mahalanobis_centers1pca0.9_bs128_epochs50_0.005_Falseba_bestusinglast_fpr_at_recall',
# 'T42_hl_llama_7B_trivia_qa_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer_supconv2_hallu_pos_0.05_bs256_epochs500_5e-05_Falseba_fpr_at_recall',
# 'T42_hl_llama_7B_trivia_qa_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer_supconv2_kmeans_hallu_pos_0.3_mahalanobis_centers1pca0.9_bs256_epochs500_5e-05_Falseba_bestusinglast_fpr_at_recall',
# 'T42_hl_llama_7B_trivia_qa_sampledplus_responses_train5000_55000_1_layerFalse_answer_last_transformer_supconv2_pos_hallu_pos_0.05_bs352_epochs500_0.0005_Falseba_fpr_at_recall',
# 'T42_hl_llama_7B_trivia_qa_sampledplus_responses_train5000_55000_1_layerFalse_answer_last_transformer_supconv2_pos_kmeans_hallu_pos_mahalanobis_centers1pca0.9_bs352_epochs500_0.005_Falseba_bestusinglast_fpr_at_recall',
# 'T42_hl_llama_7B_trivia_qa_sampledplus_responses_train5000_55000_1_layerFalse_answer_last_transformer_supconv2_pos_wp_hallu_pos_0.3_bs352_epochs500_5e-05_Falseba_fpr_at_recall',
# 'T42_hl_llama_7B_trivia_qa_sampledplus_responses_train5000_55000_1_layerFalse_answer_last_transformer_supconv2_pos_wp_kmeans_hallu_pos_2.0_mahalanobis_centers1pca0.9_bs352_epochs500_0.0005_Falseba_bestusinglast_fpr_at_recall'
# ]
# labels = ['linear_last_layer','tfmr_all_layers','tfmr_all_layers_dist','tfmr_supcon','tfmr_supcon_dist','tfmr_supcon+','tfmr_supcon+_dist','tfmr_supcon+*','tfmr_supcon+*_dist']
# paths = ['T42_hl_llama_7B_trivia_qa_sampledplus_responses_train5000_55000_1_layerFalse_answer_last_transformer_supconv2_pos_wp_hallu_pos_0.3_bs352_epochs500_5e-05_Falseba_fpr_at_recall',
# 'T42_hl_llama_7B_trivia_qa_sampledplus_responses_train5000_55000_1_layerFalse_answer_last_transformer_supconv2_pos_wp_kmeans_hallu_pos_2.0_mahalanobis_centers1pca0.9_bs352_epochs500_0.0005_Falseba_bestusinglast_fpr_at_recall',
# 'T42_hl_llama_7B_trivia_qa_sampledplus_responses_train5000_55000_1_layerFalse_answer_last_transformer_supconv2_pos_wp_hallu_pos_0.3_0.5_0.5_bs352_epochs500_5e-05_Falseba',

# ]
# labels = ['pos_wp','pos_wp_dist','0.5pos_0.5wp']
## hl-nq
paths = ['NLSC42_hl_llama_7B_nq_open_greedy_responses_train5000_5000_1_layerFalse_answer_last_individual_linear_hallu_pos_bs128_epochs50_0.005_False_fpr_at_recall',
'T42_hl_llama_7B_nq_open_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer_hallu_pos_bs128_epochs50_5e-05_Falseba_fpr_at_recall',
'T42_hl_llama_7B_nq_open_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer_kmeans_hallu_pos_mahalanobis_centers1pca0.9_bs128_epochs50_0.005_Falseba_bestusinglast_fpr_at_recall',
'T42_hl_llama_7B_nq_open_greedy_responses_train5000_5000_1_layerFalse_answer_last_transformer_supconv2_hallu_pos_0.2_bs256_epochs500_0.0005_Falseba_fpr_at_recall',

]
labels = []
for path,label in zip(paths,labels):
    recall_vals = np.load(f'{save_path}/fpr_at_recall_curves/{path}_xaxis.npy')
    fpr_at_recall_vals = np.load(f'{save_path}/fpr_at_recall_curves/{path}_yaxis.npy')
    axs.plot(recall_vals,fpr_at_recall_vals,marker='o',label=label)
    # for xy in zip(recall_vals,fpr_at_recall_vals):
    #     axs.annotate('(%.2f, %.2f)' % xy, xy=xy)
axs.legend()    
axs.set_xlabel('Recall')
axs.set_ylabel('FPR')
axs.title.set_text('FPR at recall')
fig.savefig(f'{save_path}/fpr_at_recall_curves/llama_trivia_fpr_at_recall.png')
# fig.savefig(f'{save_path}/fpr_at_recall_curves/llama_trivia_lwgting_fpr_at_recall.png')