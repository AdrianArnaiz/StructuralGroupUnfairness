import networkx as nx
import numpy as np
import pandas as pd

import utils.resistance_metrics as ermet
import utils.link_addition as rew
import utils.vis as vis

import utils.data_loader as loader

from karateclub import DeepWalk

from tqdm import tqdm
import time
import os.path as osp
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   
from matplotlib.ticker import FormatStrFormatter


N_LINKS = 10
dataset = "SBM"

DEEPWALK_baseline = True
SAVE_GRAPH = True
DRAW_GRAPH = True
DRAW_GRAPH_COMPARISON = True

print(f"[{time.strftime('%d_%m_%y__%H_%M_%S')}] - Create Folders")
exp_time = time.strftime('%d_%m_%y__%H_%M_%S')
output_path_folder = osp.join(osp.dirname(osp.realpath(__file__)), 'results', dataset+"_"+str(N_LINKS)+"lnks_"+exp_time)
os.mkdir(output_path_folder)
os.mkdir(osp.join(output_path_folder, 'metrics'))
os.mkdir(osp.join(output_path_folder, 'figs'))

if SAVE_GRAPH:
    os.mkdir(osp.join(output_path_folder, 'graphs'))

##################################################
# Load graph
print(f"[{time.strftime('%d_%m_%y__%H_%M_%S')}] - Load Graph {dataset}")
if dataset == "facebook":
    adj, _, _, _, _, _, sens = loader.process_facebook('data/facebook')
    # Depending Netx version
    #FB = nx.from_numpy_matrix(adj)
    FB = nx.from_numpy_array(adj)
    FB = nx.relabel_nodes(FB, {node:ix for ix, node in enumerate(FB.nodes())}, copy=True)

    Gcc = sorted(nx.connected_components(FB), key=len, reverse=True)
    G = nx.Graph(FB.subgraph(Gcc[0]).copy())

    sensitive_group = sens[np.array(list(Gcc[0]))]

    index_to_node = {index: node for index, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, {node:ix for ix, node in enumerate(G.nodes())}, copy=True)

elif dataset == "cat":
    G = nx.Graph()
    G.add_edges_from([(0,1),(0,3),(1,2),(1,3),(1,4),(2,4),(2,11),(3,4),(3,5),
                  (4,5),(5,6),(5,7),(6,7),(6,8),(6,9), (7,9),(7,10),(8,9),
                  (9,10),(10,15),(11,12),(12,13),(13,14),(14,15),(0,11),(8,15)])
    sensitive_group = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0])

elif dataset == "twitter":
    pass
elif dataset == "emailarenas":
    pass
elif dataset == "pokecz":
    df = pd.read_csv(f'data{os.sep}pokec_regions{os.sep}region_job.csv', index_col=0)
    pokek = nx.read_edgelist(f'data{os.sep}pokec_regions{os.sep}region_job_relationship.txt')
    pokek = nx.relabel_nodes(pokek, {node:int(node) for node in pokek.nodes()}, copy=True) #str to int
    nx.set_node_attributes(pokek, dict(df['gender']), name='sens')
    nx.set_node_attributes(pokek, dict(df['region']), name='region')

    Gcc = sorted(nx.connected_components(pokek), key=len, reverse=True)
    G = nx.Graph(pokek.subgraph(Gcc[0]).copy())
    del pokek
    del df
    G = nx.relabel_nodes(G, {node:ix for ix, node in enumerate(G.nodes())}, copy=True) # from 0 to n: simplify adding edges

    sensitive_group = nx.get_node_attributes(G, 'sens')
    # sensitive group to numpy array
    sensitive_group = np.array(list(sensitive_group.values()))

elif dataset == "pokecn":
    df = pd.read_csv(f'data{os.sep}pokec_regions{os.sep}region_job_2.csv', index_col=0)
    pokek = nx.read_edgelist(f'data{os.sep}pokec_regions{os.sep}region_job_2_relationship.txt')
    pokek = nx.relabel_nodes(pokek, {node:int(node) for node in pokek.nodes()}, copy=True) #str to int
    nx.set_node_attributes(pokek, dict(df['gender']), name='sens')
    nx.set_node_attributes(pokek, dict(df['region']), name='region')

    Gcc = sorted(nx.connected_components(pokek), key=len, reverse=True)
    G = nx.Graph(pokek.subgraph(Gcc[0]).copy())
    del pokek
    del df
    G = nx.relabel_nodes(G, {node:ix for ix, node in enumerate(G.nodes())}, copy=True) # from 0 to n: simplify adding edges

    sensitive_group = nx.get_node_attributes(G, 'sens')
    # sensitive group to numpy array
    sensitive_group = np.array(list(sensitive_group.values()))
elif dataset == "SBM":
    sizes = [40, 40, 15, 15]
    sizes = [150, 150, 50, 50]
    c1 = 0.8
    c2 = 0.8
    c3 = 0.8 
    c4 = 0.95 
    c1_c2 = 0.03
    c1_c3 = 0.01
    c2_c3 = 0.01
    c1_c4 = 0.00
    c2_c4 = 0.00
    c3_c4 = 0.02

    group_by_com = [0,0,1,0]
    noises = [0,0,0,0]
    #noises = [0,0,0,0]
    probs = [[c1, c1_c2, c1_c3, c1_c4],
            [c1_c2, c2, c2_c3, c2_c4],
            [c1_c3, c2_c3, c3, c3_c4],
            [c1_c4, c2_c4, c3_c4, c4]]
    G = nx.stochastic_block_model(sizes, probs)#, seed=0)

    sensitive_group = np.array([])
    for ix, s in enumerate(sizes):
        s_c = np.array([group_by_com[ix]]*s)
        if noises[ix] !=0:
            idx = np.random.choice(s,int(s*noises[ix]),replace=False)
            idx = np.random.choice(range(s)) if len(idx)==0 else idx
            s_c[idx] = 1 - s_c[idx]
        sensitive_group = np.hstack([sensitive_group, s_c])

elif dataset == "karate":
    G = nx.karate_club_graph()
    sensitive_group = np.ones(len(G.nodes()))
    sensitive_group[-4:] = 0

print(f"[{time.strftime('%d_%m_%y__%H_%M_%S')}] - compute original effective resistance")
R = ermet.effective_resistance_matrix(G)

print(f"[{time.strftime('%d_%m_%y__%H_%M_%S')}] - compute initial metrics based on R")
# Get initial metrics
init_total= ermet.group_total_reff(R, sensitive_group)
init_diam= ermet.group_reff_diam(R, sensitive_group)
init_avg_diam = ermet.group_avg_reff_diam(R, sensitive_group)
init_max_bet = ermet.group_max_reff_betweeness(R, G, sensitive_group)
init_avg_bet = ermet.group_avg_reff_betweeness(R, G, sensitive_group)
init_std_bet = ermet.group_std_reff_betweeness(R, G, sensitive_group)

ori_metric_dict = {'total':init_total,
                   'diam':init_diam,
                   'avg_diam':init_avg_diam,
                   'max_bet' :init_max_bet,
                   'avg_bet':init_avg_bet,
                   'std_bet':init_std_bet}

print(f"[{time.strftime('%d_%m_%y__%H_%M_%S')}] - Add links")
Graphs, data = rew.add_links(G, N_LINKS, sensitive_group)
G_s, G_w, G_aw, G_as = Graphs
d_s, d_w, d_aw, d_as = data

assert G.number_of_edges()+1+N_LINKS==G_w.number_of_edges()
assert G_s.number_of_edges()==G_w.number_of_edges()==G_aw.number_of_edges()==G_as.number_of_edges()

print(f"[{time.strftime('%d_%m_%y__%H_%M_%S')}] - Add DW links")
############################################
## Compute DeepWalk baseline
if DEEPWALK_baseline:
    G_dw_s = G.copy()
    #G_dw_s = nx.relabel_nodes(G_dw_s, {node:ix for ix, node in enumerate(G_dw_s.nodes())}, copy=False)

    for i in tqdm(range(N_LINKS)):
        dw = DeepWalk(dimensions=32)
        dw.fit(G_dw_s.copy())
        Z = dw.get_embedding()
        S = np.dot(Z, Z.T)
        np.fill_diagonal(S,0)
        G_dw_s = rew.add_weak_link(G_dw_s, CT=S)
        R = ermet.effective_resistance_matrix(G_dw_s)
        
        
    G_dw_w = G.copy()
    #G_dw_w = nx.relabel_nodes(G_dw_w, {node:ix for ix, node in enumerate(G_dw_w.nodes())}, copy=False)
    for i in tqdm(range(N_LINKS)):
        dw = DeepWalk(dimensions=32)
        dw.fit(G_dw_w.copy())
        Z = dw.get_embedding()
        S = np.dot(Z, Z.T)
        np.fill_diagonal(S,0)
        G_dw_w = rew.add_strong_link(G_dw_w, CT=S)


############################################
## Save graphs
if SAVE_GRAPH:
    print(f"[{time.strftime('%d_%m_%y__%H_%M_%S')}] - Save graphs")
    if DEEPWALK_baseline:
        graph_list = [G, G_dw_s, G_s, G_as, G_dw_w, G_w, G_aw]
        names=['Orig','St-DW','Strong','St-AA', 'W-DW','ER-L','ERA-L']
    else:
        graph_list = [G, G_s, G_as, G_w, G_aw]
        names=['Orig' ,'Strong', 'St-AA','ER-L','ERA-L']

    G_edges_ori = set(G.edges())
    for i, gg in enumerate(graph_list):
        #Add attribute in the new edges (edges different to the original graph G)
        G_edge_diff = G_edges_ori ^ gg.edges()
        d_ori = dict(zip((G_edges_ori), [0]*(len(G_edges_ori))))
        d_new = dict(zip((G_edge_diff), [1]*(len(G_edge_diff))))
        nx.set_edge_attributes(gg, d_ori, name='new')
        nx.set_edge_attributes(gg, d_new, name='new')

        if 'sens' not in G.nodes[0].keys():
            d_sens = dict(zip(gg.nodes(), sensitive_group))
            nx.set_node_attributes(gg, d_sens, name='sens')

        #nx.write_gpickle(gg, f'{output_path_folder}{os.sep}graphs{os.sep}{dataset}-{names[i]}.gpickle')
        nx.write_gexf(gg, f'{output_path_folder}{os.sep}graphs{os.sep}{dataset}-{names[i]}.gexf')


############################################
# Draw original graph

if DRAW_GRAPH:
    print(f"[{time.strftime('%d_%m_%y__%H_%M_%S')}] - Draw graphs")
    pos= nx.kamada_kawai_layout(G)
    f, axs = plt.subplots(1, figsize=(5,4))
    options = {
        "node_color": sensitive_group,
        "edge_color": 'grey',
        "width": 1,
        "node_size":(np.log(np.array([itm[1] for itm in dict(G.degree).items()]))+1)*12,
        #"node_size":20,
        "edge_cmap": plt.cm.seismic,
        "cmap": plt.cm.seismic,
        "with_labels": False,
    }
    nx.draw(G, pos, **options,ax=axs)
    f.savefig(f'{output_path_folder}{os.sep}figs{os.sep}{dataset}-original-graph.pdf', dpi=300, bbox_inches='tight', transparent=True)




############################################
## Get evolution plots
print(f"[{time.strftime('%d_%m_%y__%H_%M_%S')}] - Save evolution metrics plots")
methods = ['strong', 'ER-L', 'ERA-L', 'St-AA']
colors = {'strong': 'red', 'ER-L':'blue', 'ERA-L':'green','St-AA':'orange'}
metrics = ['total', 'diam', 'avg_diam', 'max_bet', 'avg_bet', 'std_bet']
result_dict = dict(zip(methods,data))

metric_name = {'total':'$Res_G(S)$',
               'diam': '$\mathcal{R}_{diam}(S)$',
               'avg_diam':'$\overline{\mathcal{R}_{diam}(S)}$',
               'max_bet':'Max $\mathsf{B_R}(S)$',
               'avg_bet':'$\mathsf{B_R}(S)$',
               'std_bet':'Var$(\mathsf{B_R}(S))$'
}

for metric in metrics:
    f, axs = plt.subplots(1, figsize=(3,2))
    for alg in methods:
        if alg=='ER-L' or alg=='strong':
            lw = 1.2
            ls = "dotted"
        else:
            lw = 1.4
            ls="dashdot"
        #plt.plot([d[0] for d in result_dict[alg][metric]],linewidth=lw, color=colors[alg], linestyle="-")
        plt.plot([d[1] for d in result_dict[alg][metric]], linewidth=lw, color=colors[alg], linestyle=ls)
        axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs.tick_params(axis='both', which='major', labelsize=6)
        
    legend_elements = [Line2D([0], [0], linestyle='-', color='black', label=r'$Res_G(Oth)$',linewidth='2'),
                   Line2D([0], [0], linestyle='--', color='black', label='$Res_G(Vuln)$',linewidth='2'),
                   Line2D([0], [0], linestyle='-', color='red', label='Common',linewidth='5'),
                   Line2D([0], [0], linestyle='-', color='orange', label='SAA',linewidth='5'),
                   Line2D([0], [0], linestyle='-', color='blue', label='Weak',linewidth='5'),
                   Line2D([0], [0], linestyle='-', color='green', label='AAA',linewidth='5')
                  ]
    legend_elements = [Line2D([0], [0], linestyle='dashdot', color='green', label=r'ERA-L',linewidth='2'),
                   Line2D([0], [0], linestyle='dashdot', color='orange', label='St-AA',linewidth='2'),
                   Line2D([0], [0], linestyle='dotted', color='red', label='Strong',linewidth='2'),
                       Line2D([0], [0], linestyle='dotted', color='blue', label='ER-L',linewidth='2')
                  ]
    
    #plt.legend(handles=legend_elements, bbox_to_anchor=(1.0, 0.9), loc="upper left", fontsize=9)
    plt.xlabel('# of links added')
    plt.title(metric_name[metric])
    plt.tight_layout()
    f.savefig(f'{output_path_folder}{os.sep}figs{os.sep}{dataset}-{metric}Evolution.pdf', dpi=300, bbox_inches='tight', transparent=True)

############################################
## Plot rewired graphs

if DRAW_GRAPH_COMPARISON:
    print(f"[{time.strftime('%d_%m_%y__%H_%M_%S')}] - Draw rewired graphs")
    if DEEPWALK_baseline:
        graph_list = [G, G_dw_s, G_s, G_as, G_dw_w, G_w, G_aw]
        names=['Orig','St-DW','Strong','St-AA', 'W-DW','ER-L','ERA-L']
    else:
        graph_list = [G, G_s, G_as, G_w, G_aw]
        names=['Ori' ,'Strong', 'St-AA','ER-L','ERA-L']
    
    f = vis.print_several_graphs(graph_list,
                        node_color=sensitive_group,
                        base_G=0,
                        names=names,
                        node_size=0.2,
                        pos=pos,
                        show_plot=False)
    f.savefig(f'{output_path_folder}{os.sep}figs{os.sep}{dataset}-rewired-graphs.pdf', dpi=300, bbox_inches='tight', transparent=True)


    f = vis.compare_graphs(graph_list,
                       graph_names=names, 
                       edge_highlight=True, node_size=0.2,
                       save=False, show_plot=False, return_fig=True)
    f.savefig(f'{output_path_folder}{os.sep}figs{os.sep}{dataset}-rewired-graphs-metrics.pdf', dpi=300, bbox_inches='tight', transparent=True)



#########################################
## Violinplots
print(f"[{time.strftime('%d_%m_%y__%H_%M_%S')}] - Draw violinplots last metrics")
f,_ = vis.plot_violins_node_metrics_by_group(graph_list,
                                             names=names,
                                             S=sensitive_group,
                                             orient='h',
                                             plot_max_avg_lines=True,
                                             fig_size=(12,4), show_plot=False)
f.savefig(f'{output_path_folder}{os.sep}figs{os.sep}{dataset}-violinplots.pdf', dpi=300, bbox_inches='tight', transparent=True)


############################################
## Get metric tables
print(f"[{time.strftime('%d_%m_%y__%H_%M_%S')}] - Get metric tables")
#get last iteration from effres based methods
final_result_dict = {}

for a in result_dict:
    final_result_dict[a]={}
    for m in result_dict[a]:
        final_result_dict[a][m]={}
        final_result_dict[a][m][0]=result_dict[a][m][-1][0]
        final_result_dict[a][m][1]=result_dict[a][m][-1][1]

orig_metric_dict = {}
if DEEPWALK_baseline:
    orig_metric_dict['ST-DW']={}
    orig_metric_dict['w-DW']={}
for a in result_dict:
    orig_metric_dict[a]={}
    for m in result_dict[a]:
        orig_metric_dict[a][m]={}
        orig_metric_dict[a][m][0]=result_dict[a][m][0][0]
        orig_metric_dict[a][m][1]=result_dict[a][m][0][1]
        if DEEPWALK_baseline:
            orig_metric_dict['ST-DW'][m]=orig_metric_dict['w-DW'][m]={}
            orig_metric_dict['ST-DW'][m][0]=orig_metric_dict['w-DW'][m][0]=result_dict[a][m][0][0]
            orig_metric_dict['ST-DW'][m][1]=orig_metric_dict['w-DW'][m][1]=result_dict[a][m][0][1]


final_metric_df = pd.DataFrame.from_dict({(i,j): final_result_dict[i][j] 
                           for i in final_result_dict.keys() 
                           for j in final_result_dict[i].keys()},
                       orient='index')
orig_metric_df = pd.DataFrame.from_dict({(i,j): orig_metric_dict[i][j] 
                           for i in orig_metric_dict.keys() 
                           for j in orig_metric_dict[i].keys()},
                       orient='index')

#get DW final metrics
if DEEPWALK_baseline:
    R = ermet.effective_resistance_matrix(G_dw_s)
    final_metric_df.loc[('ST-DW','total'), 0] = ermet.group_total_reff(R, sensitive_group)[0]
    final_metric_df.loc[('ST-DW','total'), 1] = ermet.group_total_reff(R, sensitive_group)[1]
    final_metric_df.loc[('ST-DW','diam'), 0] = ermet.group_reff_diam(R, sensitive_group)[0]
    final_metric_df.loc[('ST-DW','diam'), 1] = ermet.group_reff_diam(R, sensitive_group)[1]
    final_metric_df.loc[('ST-DW','avg_bet'), 0] = ermet.group_avg_reff_betweeness(R,
                                                                                G_dw_s, 
                                                                                sensitive_group)[0]
    final_metric_df.loc[('ST-DW','avg_bet'), 1] = ermet.group_avg_reff_betweeness(R,
                                                                                G_dw_s,
                                                                                sensitive_group)[1]
    final_metric_df.loc[('ST-DW','std_bet'), 0] = ermet.group_std_reff_betweeness(R,
                                                                                G_dw_s,
                                                                                sensitive_group)[0]
    final_metric_df.loc[('ST-DW','std_bet'), 1] = ermet.group_std_reff_betweeness(R,
                                                                                G_dw_s,
                                                                                sensitive_group)[1]

    R = ermet.effective_resistance_matrix(G_dw_w)
    final_metric_df.loc[('w-DW','total'), 0] = ermet.group_total_reff(R, sensitive_group)[0]
    final_metric_df.loc[('w-DW','total'), 1] = ermet.group_total_reff(R, sensitive_group)[1]
    final_metric_df.loc[('w-DW','diam'), 0] = ermet.group_reff_diam(R, sensitive_group)[0]
    final_metric_df.loc[('w-DW','diam'), 1] = ermet.group_reff_diam(R, sensitive_group)[1]
    final_metric_df.loc[('w-DW','avg_bet'), 0] = ermet.group_avg_reff_betweeness(R,
                                                                                G_dw_w, 
                                                                                sensitive_group)[0]
    final_metric_df.loc[('w-DW','avg_bet'), 1] = ermet.group_avg_reff_betweeness(R,
                                                                                G_dw_w,
                                                                                sensitive_group)[1]
    final_metric_df.loc[('w-DW','std_bet'), 0] = ermet.group_std_reff_betweeness(R,
                                                                                G_dw_w,
                                                                                sensitive_group)[0]
    final_metric_df.loc[('w-DW','std_bet'), 1] = ermet.group_std_reff_betweeness(R,
                                                                                G_dw_w,
                                                                                sensitive_group)[1]

orig_metric_df.to_csv(f'{output_path_folder}{os.sep}metrics{os.sep}Original_metrics.csv')
final_metric_df.to_csv(f'{output_path_folder}{os.sep}metrics{os.sep}Final_metrics.csv')
models = ['ST-DW','strong','St-AA','w-DW','ER-L','ERA-L'] if DEEPWALK_baseline else ['strong','St-AA','ER-L','ERA-L']
table = final_metric_df -orig_metric_df

# Absolute improvement
table = table[np.in1d(table.index.get_level_values(1), ['total', 'diam', 'avg_bet', 'std_bet'])]
# Discriminated group
group1_df = pd.DataFrame(table[1]).reset_index().pivot(index='level_0', columns='level_1')[1]
group1_df = group1_df.loc[models]
group1_df[['total', 'diam', 'avg_bet', 'std_bet']].to_csv(f'{output_path_folder}{os.sep}metrics{os.sep}Improvement_G1.csv')
# General group
group0_df = pd.DataFrame(table[0]).reset_index().pivot(index='level_0', columns='level_1')[0]
group0_df = group0_df.loc[models]
group0_df[['total', 'diam', 'avg_bet', 'std_bet']].to_csv(f'{output_path_folder}{os.sep}metrics{os.sep}Improvement_G0.csv')

# Relative improvement
table = ((final_metric_df -orig_metric_df)/orig_metric_df)*100
# Discriminated group
group1_df = pd.DataFrame(table[1]).reset_index().pivot(index='level_0', columns='level_1')[1]
group1_df = group1_df.loc[models]
group1_df[['total', 'diam', 'avg_bet', 'std_bet']].to_csv(f'{output_path_folder}{os.sep}metrics{os.sep}Improvement_relative_G1.csv')
# General group
group0_df = pd.DataFrame(table[0]).reset_index().pivot(index='level_0', columns='level_1')[0]
group0_df = group0_df.loc[models]
group0_df[['total', 'diam', 'avg_bet', 'std_bet']].to_csv(f'{output_path_folder}{os.sep}metrics{os.sep}Improvement_relative_G0.csv')


