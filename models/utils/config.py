import os
from .logger import Logger
import traceback
import sys

L = Logger()
logger = L.get_logger()

DEFAULT_CONFIG_FILE = 'default.cfg'

# configuration for FedAvg
class Config():
    def __init__(self, config_file = 'default.cfg'):
        self.config_name = config_file
        self.dataset = 'femnist'
        self.model = 'stacked_lstm'
        self.num_rounds = -1            # -1 for unlimited
        self.lr = 0.1
        self.eval_every = 3             # -1 for eval when quit
        self.clients_per_round = 10
        self.min_selected = 10
        self.max_sample = 100           # max sample num for training in a round
        self.batch_size = 10
        self.seed = 0
        self.num_epochs = 1

        self.minibatch = None       # always None for FedAvg
        self.round_ddl = 1000
        self.update_frac = 0.5
        self.max_client_num = 1000    # total client num, -1 for unlimited

        self.aggregate_algorithm = 'FedAvg'
        self.time_window = [20.0, 0.0]  # time window for selection stage
        self.user_trace = False
        self.behav_hete = False
        self.hard_hete = False

        self.no_training = False
        self.fedprox = False
        self.fedprox_mu = 0
        
        self.fedbalancer = False
        self.fb_w = 1

        self.fb_p = 0.0

        self.fb_inference_pipelining = False

        self.noise_factor = 0.0

        self.fb_simple_control_lt_stepsize = 0
        self.fb_simple_control_ddl_stepsize = 0

        self.fb_client_selection = False

        self.oort_pacer = False
        self.oort_pacer_delta = 10
        self.oort_blacklist = False
        self.oort_blacklist_rounds = 10

        self.oort = False
        self.oortbalancer = False

        self.ddl_baseline_smartpc = False
        self.ddl_baseline_smartpc_percentage = 0.0
        self.ddl_baseline_fixed = False
        self.ddl_baseline_fixed_value_multiplied_at_mean = 0.0 

        self.global_final_time = 0
        self.global_final_test_accuracy = 0.0

        self.ss_baseline = False
        
        logger.info('read config from {}'.format(config_file))
        self.read_config(config_file)
        self.log_config()
        
    def read_config(self, filename = DEFAULT_CONFIG_FILE):
        if not os.path.exists(filename):
            logger.error('ERROR: config file {} does not exist!'.format(filename))
            assert False
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                try:
                    line = line.strip().split()
                    if line[0] == 'num_rounds':
                        self.num_rounds = int(line[1])
                    elif line[0] == 'learning_rate':
                        self.lr = float(line[1])
                    elif line[0] == 'eval_every':
                        self.eval_every = int(line[1])
                    elif line[0] == 'clients_per_round':
                        self.clients_per_round = int(line[1])
                    elif line[0] == 'max_client_num':
                        self.max_client_num = int(line[1])
                        if self.max_client_num < 0:
                            self.max_client_num = sys.maxsize
                    elif line[0] == 'min_selected':
                        self.min_selected = int(line[1])
                    elif line[0] == 'batch_size':
                        self.batch_size = int(line[1])
                    elif line[0] == 'seed':
                        self.seed = int(line[1])
                    elif line[0] == 'num_epochs':
                        self.num_epochs = int(line[1])
                    elif line[0] == 'dataset':
                        self.dataset = str(line[1])
                    elif line[0] == 'model':
                        self.model = str(line[1])
                    elif line[0] == 'round_ddl':
                        self.round_ddl = float(line[1])
                    elif line[0] == 'update_frac':
                        self.update_frac = float(line[1])
                    elif line[0] == 'aggregate_algorithm':
                        self.aggregate_algorithm = str(line[1])
                    elif line[0] == 'time_window':
                        self.time_window = [float(line[1]), float(line[2])]
                    elif line[0] == 'behav_hete' :
                        self.behav_hete = line[1].strip() == 'True'
                        if not self.behav_hete:
                            logger.info('no behavior heterogeneity! assume client is availiable at any time.')
                    elif line[0] == 'hard_hete' :
                        self.hard_hete = line[1].strip() == 'True'
                        if not self.hard_hete:
                            logger.info('no hardware heterogeneity! assume all clients are same.')
                    elif line[0] == 'max_sample' :
                        self.max_sample = int(line[1])
                    elif line[0] == 'fedprox':
                        self.fedprox = line[1].strip()=='True'
                    elif line[0] == 'fedprox_mu':
                        self.fedprox_mu = float(line[1].strip())
                    elif line[0] == 'user_trace':
                        # to be compatibale with old version
                        self.user_trace = line[1].strip()=='True'
                    elif line[0] == 'fedbalancer':
                        self.fedbalancer = line[1].strip()=='True'
                    elif line[0] == 'fb_w':
                        self.fb_w = int(line[1].strip())
                    elif line[0] == 'fb_client_selection':
                        self.fb_client_selection = line[1].strip()=='True'
                    elif line[0] == 'oort':
                        self.oort = line[1].strip()=='True'
                    elif line[0] == 'oortbalancer':
                        self.oortbalancer = line[1].strip()=='True'
                    elif line[0] == 'oort_pacer':
                        self.oort_pacer = line[1].strip()=='True'
                    elif line[0] == 'oort_pacer_delta':
                        self.oort_pacer_delta = int(line[1].strip())
                    elif line[0] == 'oort_blacklist':
                        self.oort_blacklist = line[1].strip()=='True'
                    elif line[0] == 'oort_blacklist_rounds':
                        self.oort_blacklist_rounds = int(line[1].strip())
                    elif line[0] == 'fb_p':
                        self.fb_p = 1.0 - float(line[1].strip())
                    elif line[0] == 'fb_inference_pipelining':
                        self.fb_inference_pipelining = line[1].strip()=='True'
                    elif line[0] == 'fb_simple_control_lt':
                        self.fb_simple_control_lt = line[1].strip()=='True'
                    elif line[0] == 'fb_simple_control_ddl':
                        self.fb_simple_control_ddl = line[1].strip()=='True'
                    elif line[0] == 'fb_simple_control_lt_stepsize':
                        self.fb_simple_control_lt_stepsize = float(line[1].strip())
                    elif line[0] == 'fb_simple_control_ddl_stepsize':
                        self.fb_simple_control_ddl_stepsize = float(line[1].strip())
                    elif line[0] == 'ddl_baseline_smartpc':
                        self.ddl_baseline_smartpc = line[1].strip()=='True'
                    elif line[0] == 'ddl_baseline_smartpc_percentage':
                        self.ddl_baseline_smartpc_percentage = float(line[1].strip())
                    elif line[0] == 'ddl_baseline_fixed':
                        self.ddl_baseline_fixed = line[1].strip()=='True'
                    elif line[0] == 'ddl_baseline_fixed_value_multiplied_at_mean':
                        self.ddl_baseline_fixed_value_multiplied_at_mean = float(line[1].strip())
                    elif line[0] == 'global_final_time':
                        self.global_final_time = int(line[1].strip())
                    elif line[0] == 'global_final_test_accuracy':
                        self.global_final_test_accuracy = float(line[1].strip())
                    elif line[0] == 'noise_factor':
                        self.noise_factor = float(line[1].strip())
                    elif line[0] == 'ss_baseline':
                        self.ss_baseline = line[1].strip()=='True'
                except Exception as e:
                    traceback.print_exc()
        if self.user_trace == True:
            self.hard_hete = True
            self.behav_hete = True
    
    def log_config(self):
        configs = vars(self)
        logger.info('================= Config =================')
        for key in configs.keys():
            logger.info('\t{} = {}'.format(key, configs[key]))
        logger.info('================= ====== =================')
        
