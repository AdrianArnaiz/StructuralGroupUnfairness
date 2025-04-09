
import numpy as np
import torch

import pickle
import logging
from tqdm import tqdm
import time
import os
import argparse
import os.path as osp

import utils.link_addition_torch as lkadd
import utils.torch_resistance_metrics as ermet
import utils.data_loader as loader

#create argument parser and parse arguments

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--links', type=int, default=20,
                    help='number of links to use in the experiment')
parser.add_argument('--dataset', type=str, default='facebook',
                    help='name of the dataset to use in the experiment',
                    choices=['facebook', 'google', 'UNC28'])
parser.add_argument('--model', type=str, default='ERP', choices=['ERP', 'deepwalk', 'node2vec', 'random', 'cosine'],
                    help='name of the model to use in the experiment')
parser.add_argument('--strategy', type=str, default='weak',
                    help='weak or strong link addition strategy', choices=['weak', 'strong'])
parser.add_argument('--device', type=str, default='cpu',
                    help='name of the device to use in the experiment')
parser.add_argument('--floattype', type=str, default='float32',
                    help='name of the dtype to use in the experiment')

args = parser.parse_args()

print()
print(args)
N_LINKS = args.links
DATASET = args.dataset
MODEL = args.model
STRATEGY = args.strategy
DEVICE = args.device
DTYPE = args.floattype

if DTYPE == 'float32':
    DTYPE = torch.float32
elif DTYPE == 'float64':
    DTYPE = torch.float64
else:
    raise ValueError("Dtype not found")


# Create folders to save results
exp_time = time.strftime('%d_%m_%y__%H_%M_%S')
output_path_folder = osp.join(osp.dirname(osp.realpath(__file__)), 'results', DATASET+"_"+MODEL+"_"+STRATEGY+'_'+str(N_LINKS)+"lnks_"+exp_time)
os.mkdir(output_path_folder)


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
                    datefmt='%H:%M:%S',
                    filemode='a',
                    filename=osp.join(output_path_folder, 'log.log'))
logger = logging.getLogger(__name__)
logger.info(f"Created Folder:{output_path_folder}")
logger.info("Starting experiment.")

# log arguments
logger.info("Running experiment with arguments:")
for arg in vars(args):
    logger.info(f"{arg}: {getattr(args, arg)}")



# Load data
logger.info("Loading data")
GW = loader.load_data(dataset = DATASET,
                      device = DEVICE, 
                      dtype = DTYPE)


opt_args = {}
if MODEL == 'ERP':
    logger.info("Generating edge scores with Effective Resistance")
elif MODEL == 'deepwalk':
    logger.info("Generating edge scores with DeepWalk")
    opt_args = {'dim': 128, 'walk_length': 40, 'window_size': 10, 'workers': 4}
elif MODEL == 'node2vec':
    logger.info("Generating edge scores with Node2Vec")
    opt_args = {'dim': 32, 'walk_length': 20, 'device': DEVICE}
elif MODEL == 'random':
    pass
elif MODEL == 'cosine':
    logger.info("Generating edge scores with Cosine Similarity")
else:
    logger.error("Model not found")
    raise ValueError("Model not found")


#* Get initial metrics
results_dict = {}
unique_groups = GW.sens.unique().cpu().detach().numpy()
for group in unique_groups:
    results_dict[group] = {}
    group_res = ermet.get_group_metrics(GW.get_effective_resistance().cpu().detach(),
                                             GW.sens.cpu().detach()==group,
                                             ~GW.edge_mask.cpu().detach())
    for metric in group_res:
        results_dict[group][metric] = [group_res[metric]]

print('Initial metrics:')
print(results_dict)

#* Link addition loop
for i in tqdm(range(N_LINKS)):
    #* Get score
    if MODEL == 'random':
        u, v = lkadd.get_random_link(GW.edge_mask)
    elif MODEL == 'ERP':
        S = GW.get_effective_resistance()
    else:
        if MODEL != 'deepwalk' or i%10 == 0:
            S = lkadd.get_edge_score(GW, MODEL, **opt_args)

    
    #* Select edge based on score
    if MODEL != 'random':
        if STRATEGY == 'weak':
            u,v = lkadd.get_weakest_link(S.cpu().detach(), GW.edge_mask.cpu().detach(), GW.num_nodes)
        elif STRATEGY == 'strong':
            u,v = lkadd.get_strongest_link(S.cpu().detach(), GW.edge_mask.cpu().detach(), GW.num_nodes)
        else:
            raise ValueError("Strategy not found")
    
    if GW.is_edge(u,v):
        raise Exception('Edge already exists')
    else:
        GW.add_link(u,v)

    logger.info(f"Added link {u},{v}")
    logger.info(f"Number of links: {GW.num_edges}")
    logger.info(f"Update metrics")
    Rcpu = GW.get_effective_resistance().cpu().detach()
    for group in unique_groups:
        group_res = ermet.get_group_metrics(Rcpu,
                                            GW.sens.cpu().detach()==group,
                                            ~GW.edge_mask.cpu().detach())
        for metric in group_res:
            results_dict[group][metric].append(group_res[metric])


#* Save dict as pickle
with open(osp.join(output_path_folder, 'results_dict.pkl'), 'wb') as f:
    pickle.dump(results_dict, f)

#* Save effective resistance matrix
np.save(osp.join(output_path_folder, 'R.npy'), GW.get_effective_resistance().cpu().detach().numpy())