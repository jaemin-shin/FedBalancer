import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config',
                    help='path to config file;',
                    type=str,
                    # required=True,
                    default='default.cfg')
    
    return parser.parse_args()
