from utils.sdrf import sdrf
import utils.data_loader as loader
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

import pickle
import logging
import time
import os
import argparse
import os.path as osp

import utils.torch_resistance_metrics as ermet
import utils.data_loader as loader


#create argument parser and parse arguments

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--links', type=int, default=20,
                    help='number of links to use in the experiment')
parser.add_argument('--dataset', type=str, default='facebook',
                    help='name of the dataset to use in the experiment',
                    choices=['facebook', 'google', 'UNC28'])
parser.add_argument('--device', type=str, default='cpu',
                    help='name of the device to use in the experiment')
parser.add_argument('--floattype', type=str, default='float32',
                    help='name of the dtype to use in the experiment')

args = parser.parse_args()

print()
print(args)
N_LINKS = args.links
DATASET = args.dataset
MODEL = 'SDRF'
STRATEGY = 'weak'
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
GW = loader.load_data(dataset = DATASET, device = DEVICE,  dtype = DTYPE)
GW.mode = 'woodbury'

# Get initial metrics
results_dict = {}
unique_groups = GW.sens.unique().cpu().detach().numpy()
for group in unique_groups:
    results_dict[group] = {}
    group_res = ermet.get_group_metrics(GW.get_effective_resistance().cpu().detach(),
                                             GW.sens.cpu().detach()==group,
                                             ~GW.edge_mask.cpu().detach())
    for metric in group_res:
        results_dict[group][metric] = [group_res[metric]]

#* Rewire according to SDRF
GW, added_links = sdrf(GW, loops=N_LINKS, remove_edges=False, removal_bound=2, tau=1000, is_undirected=True,
                               unique_groups=unique_groups, results_dict=results_dict)

logger.info("Saving Results.")

#! Save Result
torch.save(torch.LongTensor(added_links), osp.join(output_path_folder, "added_links.pt"))

#* Save dict as pickle
with open(osp.join(output_path_folder, 'results_dict.pkl'), 'wb') as f:
    pickle.dump(results_dict, f)

#* Save effective resistance matrix
np.save(osp.join(output_path_folder, 'R.npy'), GW.get_effective_resistance().cpu().detach().numpy())