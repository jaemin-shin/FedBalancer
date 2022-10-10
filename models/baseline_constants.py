"""dict: Specifies execution parameters (tot_num_rounds, eval_every_num_rounds, clients_per_round)"""

MODEL_PARAMS = {
    # dataset,model_name,lr,num_classes
    'femnist.client_model': ('femnist', 'cnn', 0.001, 62), # lr, num_classes
}
"""dict: Model specific parameter specification"""

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
