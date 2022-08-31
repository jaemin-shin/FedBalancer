"""dict: Specifies execution parameters (tot_num_rounds, eval_every_num_rounds, clients_per_round)"""

MODEL_PARAMS = {
    # dataset,model_name,lr,num_classes
    'femnist.client_model': ('femnist', 'cnn', 0.001, 62), # lr, num_classes
    'har.client_model': ('har', 'cnn', 0.005, 6), # lr, num_classes
    'shakespeare.client_model': ('shakespeare', 'stacked_lstm', 0.8, 80, 80, 256), # lr, seq_len, num_classes, num_hidden
    'celeba.client_model': ('celeba', 'cnn', 0.001, 2), # lr, num_classes
    'big_reddit.client_model': ('big_reddit', 'topk_stacked_lstm', 2, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
}
"""dict: Model specific parameter specification"""

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
