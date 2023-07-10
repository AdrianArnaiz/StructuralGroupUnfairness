import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from . import resistance_metrics as ermet
from scipy.spatial.distance import squareform
import pandas as pd
import networkx as nx
from .  import spectral as spec

def plot_violins_node_metrics(df_res, df_total, df_tot_filter, df_diam, save=False):
    """_summary_

    Args:
        df_res (_type_): _description_
        df_total (_type_): _description_
        df_tot_filter (_type_): _description_
        df_diam (_type_): _description_
        save (bool, optional): _description_. Defaults to False.
    """
    f, axs = plt.subplots(2,2, figsize=(6,5))
    axs = axs.ravel()

    sns.violinplot(df_res, inner="box",  bw= 0.5, ax=axs[0], orient='v', palette='colorblind', cut=0, linewidth=0.7)
    sns.violinplot(df_total, inner="box",  bw= 1.2, ax=axs[1], orient='v', palette='colorblind', cut=0, linewidth=0.7)
    sns.violinplot(df_tot_filter, inner="box",  bw= 0.3, ax=axs[2], orient='v', palette='colorblind', cut=0, linewidth=0.7)
    sns.violinplot(df_diam, inner="box",  bw= 3, ax=axs[3], orient='v', palette='colorblind', cut=0, linewidth=0.7)

    axs[0].set_title('$R_{uv}$')
    axs[1].set_title('$Res_G(u)$')
    axs[2].set_title('$\mathsf{B_R}(u)$')
    axs[3].set_title('$\mathcal{R}_{diam}(u)$')

    axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=12, rotation=10)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=12, rotation=10)
    axs[2].set_xticklabels(axs[2].get_xticklabels(), fontsize=12, rotation=10)
    axs[3].set_xticklabels(axs[3].get_xticklabels(), fontsize=12, rotation=10)


    axs[0].scatter(x=range(4),y=df_res.mean(),c="black", marker='_',s=600)
    axs[1].scatter(x=range(4),y=df_total.mean(),c="black", marker='_',s=500)
    axs[2].scatter(x=range(4),y=df_tot_filter.mean(),c="black", marker='_',s=500)
    axs[3].scatter(x=range(4),y=df_diam.mean(),c="black", marker='_',s=500)

    axs[0].scatter(x=range(4),y=df_res.max(),c="black", marker='_',s=300)
    axs[1].scatter(x=range(4),y=df_total.max(),c="black", marker='_',s=300)
    axs[2].scatter(x=range(4),y=df_tot_filter.max(),c="black", marker='_',s=300)
    axs[3].scatter(x=range(4),y=df_diam.max(),c="black", marker='_',s=300)

    plt.tight_layout()
    if save:
        f.savefig('RewiredViolins.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_violins_node_metrics_by_group(Gs, names, S, 
                                       save=False, file_title='group_violins',
                                       fig_size=(6,8), orient='v',plot_max_avg_lines=True):
    
    all_res = []
    all_total_res = []
    all_filtered_res = []
    all_diam_res = []

    for G in Gs:
        res = ermet.effective_resistance_matrix(G)
        all_res.append(squareform(res.round(6)))
        all_total_res.append(ermet.node_total_er(res))
        all_filtered_res.append(ermet.node_total_er(res, G, filtered=True))
        all_diam_res.append(ermet.node_diameters(res))

    df_res = pd.DataFrame(all_res).transpose()
    df_total = pd.DataFrame(all_total_res).transpose()
    df_tot_filter = pd.DataFrame(all_filtered_res).transpose()
    df_diam = pd.DataFrame(all_diam_res).transpose()

    df_res.columns = names
    df_total.columns = names
    df_tot_filter.columns = names
    df_diam.columns = names

    df_total['S'] = S
    df_tot_filter['S'] = S
    df_diam['S'] = S

    df_total_M = df_total.melt(id_vars=['S'], value_vars=names,
                   var_name='Method', value_name='Values')
    df_tot_filter_M = df_tot_filter.melt(id_vars=['S'], value_vars=names,
                   var_name='Method', value_name='Values')
    df_diam_M = df_diam.melt(id_vars=['S'], value_vars=names,
                   var_name='Method', value_name='Values')


    f, axs = plt.subplots(1,4, figsize=fig_size, sharey=True)
    axs = axs.ravel()

    if orient == 'v':
        y='Values'
        x='Method'
    else:
        y='Method'
        x='Values'
    
    sns.violinplot(df_res, inner="box",  bw= 0.5, ax=axs[0], orient=orient, palette='colorblind', cut=0.1, linewidth=0.7)
    sns.violinplot(df_total_M, y=y, x=x, hue='S', ax=axs[1],
                    inner = 'box', split=True, orient=orient, scale="width",palette='colorblind', cut=0.1, linewidth=0.7)
    sns.violinplot(df_diam_M, y=y, x=x, hue='S', ax=axs[2],
                   inner = 'box', split=True, orient=orient, scale="width", palette='colorblind', cut=0.1, linewidth=0.7)
    sns.violinplot(df_tot_filter_M, y=y, x=x, hue='S', ax=axs[3],
                   inner = 'box', split=True, orient=orient, scale="width",palette='colorblind', cut=0.1, linewidth=0.7)

    axs[0].set_title('All $R_{uv}$')
    axs[1].set_title('$\mathsf{Res_G}(u)$')
    axs[2].set_title('$\mathcal{R}_{diam}(u)$')
    axs[3].set_title('$\mathsf{B_R}(u)$')

    #Set the x limit of the third plot between 2 chosen values
    #axs[3].set_xlim([1.99, 2.01])
    #axs[3].set_xscale('log')

    

    if orient=='v':
        #axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=12, rotation=10)
        #axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=12, rotation=10)
        #axs[2].set_xticklabels(axs[2].get_xticklabels(), fontsize=12, rotation=10)
        #axs[3].set_xticklabels(axs[3].get_xticklabels(), fontsize=12, rotation=10)
        if plot_max_avg_lines:
            n_gr = len(Gs)
            axs[0].scatter(x=range(n_gr),y=df_res.mean(),c="black", marker='_',s=600)
            axs[1].scatter(x=range(n_gr),y=df_total.loc[:, ~df_total.columns.isin(['S'])].mean(),c="black", marker='_',s=500)
            axs[3].scatter(x=range(n_gr),y=df_tot_filter.loc[:, ~df_tot_filter.columns.isin(['S'])].mean(),c="black", marker='_',s=500)
            axs[2].scatter(x=range(n_gr),y=df_diam.loc[:, ~df_diam.columns.isin(['S'])].mean(),c="black", marker='_',s=500)

            axs[0].scatter(x=range(n_gr),y=df_res.max(),c="black", marker='_',s=300)
            axs[1].scatter(x=range(n_gr),y=df_total.loc[:, ~df_total.columns.isin(['S'])].max(),c="black", marker='_',s=300)
            axs[3].scatter(x=range(n_gr),y=df_tot_filter.loc[:, ~df_tot_filter.columns.isin(['S'])].max(),c="black", marker='_',s=300)
            axs[2].scatter(x=range(n_gr),y=df_diam.loc[:, ~df_diam.columns.isin(['S'])].max(),c="black", marker='_',s=300)
    else:
        #axs[0].set_yticklabels(axs[0].get_yticklabels(), fontsize=12, rotation=10)
        #axs[1].set_yticklabels(axs[1].get_yticklabels(), fontsize=12, rotation=10)
        #axs[2].set_yticklabels(axs[2].get_yticklabels(), fontsize=12, rotation=10)
        #axs[3].set_yticklabels(axs[3].get_yticklabels(), fontsize=12, rotation=10)
        axs[0].set_ylabel('')
        axs[1].set_ylabel('')
        axs[2].set_ylabel('')
        axs[3].set_ylabel('')
        axs[0].set_xlabel('')
        axs[1].set_xlabel('')
        axs[2].set_xlabel('')
        axs[3].set_xlabel('')
        axs[1].get_legend().get_texts()[0].set_text('Others')
        axs[1].get_legend().get_texts()[1].set_text('Vulnerable')
        axs[2].get_legend().remove()
        axs[3].get_legend().remove()

        if plot_max_avg_lines:
            n_gr = len(Gs)
            axs[0].scatter(y=range(n_gr),x=df_res.mean(),c="black", marker='|',s=600)
            axs[1].scatter(y=range(n_gr),x=df_total.loc[:, ~df_total.columns.isin(['S'])].mean(),c="black", marker='|',s=500)
            axs[3].scatter(y=range(n_gr),x=df_tot_filter.loc[:, ~df_tot_filter.columns.isin(['S'])].mean(),c="black", marker='|',s=500)
            axs[2].scatter(y=range(n_gr),x=df_diam.loc[:, ~df_diam.columns.isin(['S'])].mean(),c="black", marker='|',s=500)

            axs[0].scatter(y=range(n_gr),x=df_res.max(),c="black", marker='|',s=300)
            axs[1].scatter(y=range(n_gr),x=df_total.loc[:, ~df_total.columns.isin(['S'])].max(),c="black", marker='|',s=300)
            axs[3].scatter(y=range(n_gr),x=df_tot_filter.loc[:, ~df_tot_filter.columns.isin(['S'])].max(),c="black", marker='|',s=300)
            axs[2].scatter(y=range(n_gr),x=df_diam.loc[:, ~df_diam.columns.isin(['S'])].max(),c="black", marker='|',s=300)

    plt.tight_layout()
    if save:
        f.savefig(file_title+'.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return f, axs


def plot_violins_node_metrics_by_group2(Gs, names, S, save=False, file_title='group_violins', fig_size=(6,8), orient='v'):
    
    all_res = []
    all_total_res = []
    all_diam_res = []

    for G in Gs:
        res = ermet.effective_resistance_matrix(G)
        all_res.append(squareform(res.round(5)))
        all_total_res.append(ermet.node_total_er(res))
        all_diam_res.append(ermet.node_diameters(res))

    df_res = pd.DataFrame(all_res).transpose()
    df_total = pd.DataFrame(all_total_res).transpose()
    df_diam = pd.DataFrame(all_diam_res).transpose()

    df_res.columns = names
    df_total.columns = names
    df_diam.columns = names

    df_total['S'] = S
    df_diam['S'] = S

    df_total_M = df_total.melt(id_vars=['S'], value_vars=names,
                   var_name='Method', value_name='Values')
    df_diam_M = df_diam.melt(id_vars=['S'], value_vars=names,
                   var_name='Method', value_name='Values')


    MOSAIC="""
    00
    12
    """
    f,axs_dict = plt.subplot_mosaic(MOSAIC, figsize=fig_size, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [.6, 1]})
    axs = f.get_axes()

    if orient == 'v':
        y='Values'
        x='Method'
    else:
        y='Method'
        x='Values'
    
    sns.violinplot(df_res, inner="box",  bw= 0.5, ax=axs[0], orient=orient, palette='colorblind', cut=0, linewidth=0.3)
    sns.violinplot(df_total_M, y=y, x=x, hue='S', ax=axs[1], cut=0, linewidth=0.3,
                    inner = 'box', split=False, orient=orient, scale="width",palette='colorblind')
    sns.violinplot(df_diam_M, y=y, x=x, hue='S', ax=axs[2], cut=0, linewidth=0.3,
                   inner = 'box', split=False, orient=orient, scale="width",palette='colorblind')

    axs[0].set_title('$R_{uv}$')
    axs[1].set_title('$Res_G(u)$')
    axs[2].set_title('$\mathcal{R}_{diam}(u)$')

    if orient=='v':
        axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=12, rotation=10)
        axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=12, rotation=10)
        axs[2].set_xticklabels(axs[2].get_xticklabels(), fontsize=12, rotation=10)

        n_gr = len(Gs)
        axs[0].scatter(x=range(n_gr),y=df_res.mean(),c="black", marker='_',s=300)
        axs[1].scatter(x=range(n_gr),y=df_total.loc[:, ~df_total.columns.isin(['S'])].mean(),c="black", marker='_',s=300)
        axs[2].scatter(x=range(n_gr),y=df_diam.loc[:, ~df_diam.columns.isin(['S'])].mean(),c="black", marker='_',s=300)

        axs[0].scatter(x=range(n_gr),y=df_res.max(),c="black", marker='_',s=300)
        axs[1].scatter(x=range(n_gr),y=df_total.loc[:, ~df_total.columns.isin(['S'])].max(),c="black", marker='_',s=300)
        axs[2].scatter(x=range(n_gr),y=df_diam.loc[:, ~df_diam.columns.isin(['S'])].max(),c="black", marker='_',s=300)
    elif orient=='h':
        axs[0].set_yticklabels(axs[0].get_yticklabels(), fontsize=12, rotation=10)
        axs[1].set_yticklabels(axs[1].get_yticklabels(), fontsize=12, rotation=10)
        axs[2].set_yticklabels(axs[2].get_yticklabels(), fontsize=12, rotation=10)

        n_gr = len(Gs)
        axs[0].scatter(y=range(n_gr),x=df_res.mean(),c="black", marker='|',s=300)
        axs[1].scatter(y=range(n_gr),x=df_total.loc[:, ~df_total.columns.isin(['S'])].mean(),c="black", marker='|',s=300)
        axs[2].scatter(y=range(n_gr),x=df_diam.loc[:, ~df_total.columns.isin(['S'])].mean(),c="black", marker='|',s=300)

        axs[0].scatter(y=range(n_gr),x=df_res.max(),c="black", marker='|',s=300)
        axs[1].scatter(y=range(n_gr),x=df_total.loc[:, ~df_total.columns.isin(['S'])].max(),c="black", marker='|',s=300)
        axs[2].scatter(y=range(n_gr),x=df_diam.loc[:, ~df_total.columns.isin(['S'])].max(),c="black", marker='|',s=300)

    axs[2].axes.get_yaxis().set_ticks([])
    #Remove ylabl from axs[2]
    axs[0].set_ylabel('')
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')
    axs[1].get_legend().remove()

    plt.tight_layout()
    if save:
        f.savefig(file_title+'.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return f, axs

def print_several_graphs(Gs, node_color=None, base_G=0, names=None, pos=None, colorbar=False, node_size=1, show_base = True, show_plot=True):
    num_vis_G = int(np.ceil(len(Gs)/2)) if show_base else int(np.ceil((len(Gs)-1)/2))
    f, axs = plt.subplots(num_vis_G, 2, figsize=(7,num_vis_G*4))
    #f, axs = plt.subplots(1, 4, figsize=(14,5))
    options = {
        "edge_color": 'grey',
        "width": 0.5,
        "edge_cmap": plt.cm.seismic,
        "cmap": plt.cm.seismic,
        "with_labels": False,
        'pos':pos,
        'font_color':'w'
    }
    
    axs = axs.ravel()
    aux=10
    
    ori_G = Gs[base_G]
    #ori_reff = ermet.effective_resistance_matrix(ori_G)
    
    for i, G in enumerate(Gs):
        idx = i if show_base else i-1
        G_diff_edges = list(Gs[base_G].edges() ^ G.edges())
        R = ermet.effective_resistance_matrix(G)
        
        if node_color is None:
            node_color = ermet.node_total_er(ermet.effective_resistance_matrix(G))
        elif type(node_color) == str:
            if node_color=='total':
                node_color = ermet.node_total_er(R)
            elif node_color=='between':
                node_color = ermet.node_total_er(R, G, filtered=True)
            elif node_color=='diam':
                node_color = ermet.diameters(R)
        if i==base_G:
            ori_node_color = node_color.copy()
            if colorbar:
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                    norm=plt.Normalize(vmin=np.min(ori_node_color), vmax=np.max(ori_node_color)))

                f.colorbar(sm, cax=axs[idx], orientation='vertical')
        
        if (i==base_G and show_base) or i!=base_G:
            axs[idx].set_title(names[i])
            nx.draw(G, ax = axs[idx], **options,
                node_size = np.array([d+aux for n,d in G.degree()])*node_size,
                node_color=node_color,
                #linewidths=0.1, edgecolors='white',
                vmin =ori_node_color.min() , vmax=ori_node_color.max())
            nx.draw_networkx_edges(G, edgelist = G_diff_edges, pos=pos, edge_color='orange', width=2, ax = axs[idx])

    for ax in axs:
        ax.axis('off')   
    plt.tight_layout()

    if show_plot:
        plt.show()
    return f




def graph_summary_metrics(G, pos=None):
    
    original_res = ermet.effective_resistance_matrix(G)
    evl, evc = spec.find_evecs(nx.laplacian_matrix(G))
    evl = np.real(evl)
    evc = np.real(evc)
        
    MOSAIC="""
    ..00..
    112233
    444555
    666777
    """
    f,axs_dict = plt.subplot_mosaic(MOSAIC, figsize=(12,14),
                                    gridspec_kw={'height_ratios': [1, 1.2, 1, 1]})
    axs = f.get_axes()
    
    # Draw the graph
    pos = nx.kamada_kawai_layout(G) if pos is None else pos #spring_layout(G)
    options = {
        "node_color": np.array(evc[:,1]).ravel()+20,
        "edge_color": 'grey',
        "width": 0.8,
        "node_size":np.array([itm[1] for itm in dict(G.degree).items()])*5,
        "edge_cmap": plt.cm.seismic,
        "cmap": plt.cm.seismic,
        "with_labels": False,
    }
    axs[0].set_title('Graph (color=$\lambda_2$, size=$D$)', fontsize=20)
    nx.draw(G, pos, **options, ax=axs[0])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.seismic, 
                            norm=plt.Normalize(vmin=np.min(evc[:,1]), vmax=np.max(evc[:,1])))
    plt.colorbar(sm, ax=axs[0])
    
    axs[0].set_axis_on()
    total = ermet.total_effective_resistance(original_res)
    diam = ermet.resistance_diameter(original_res)
    avg_diam = ermet.avg_node_max_distance(original_res)
    title = f"Graph Metrics: $Res_G$: {total:.2e} -- "
    title += f"$\mathcal{{R}}_{{diam}}$: {diam:.2e}"
    title += f" -- $\overline{{\mathcal{{R}}_{{diam}}(V)}}$: {avg_diam:.2e}"
    axs[0].set_xlabel(title, fontsize=13)
    
    ######
    ## Node total er as color node over the graph
    f_title = '$Res_G(u)$'
    axs[1].set_title(f_title, fontsize=20)
    nodes_total_er = ermet.node_total_er(original_res)
    options['node_size'] = 100
    options['node_color'] = nodes_total_er
    options["cmap"] = plt.cm.viridis
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                            norm=plt.Normalize(vmin=np.min(nodes_total_er), vmax=np.max(nodes_total_er)))
    cbar = plt.colorbar(sm, ax=axs[1])
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_linewidth(0)
    axs[1].set_xlabel('$Res_G(u)$', fontsize=20)
    nx.draw(G, pos, **options, ax=axs[1])
    
    ######
    ## Node betweeness
    axs[2].set_title('$Res_G(u, \mathcal{N})$', fontsize=20)
    nodes_betwn = ermet.node_total_er(original_res, G=G, filtered=True)
    options['node_color'] = nodes_betwn
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                            norm=plt.Normalize(vmin=np.min(nodes_betwn), vmax=np.max(nodes_betwn)))
    cbar = plt.colorbar(sm, ax=axs[2])
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_linewidth(0)
    nx.draw(G, pos, **options, ax=axs[2])
    
    ######
    ## Node diameters as color node over the graph
    axs[3].set_title('$\mathcal{R}_{diam}(u)$', fontsize=20)
    nodes_diams = ermet.node_diameters(original_res)
    options['node_color'] = nodes_diams
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                            norm=plt.Normalize(vmin=np.min(nodes_diams), vmax=np.max(nodes_diams)))
    cbar = plt.colorbar(sm, ax=axs[3])
    nx.draw(G, pos, **options, ax=axs[3])
    # Set the tick font size of the colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_linewidth(0)
    
    ######
    ## Tots ER
    bins = int(np.sqrt(G.number_of_edges()))
    axs[4].hist(original_res[np.triu_indices(np.sum(G.number_of_nodes()),k=1)],alpha=1,bins=bins, color='black')
    title = "All $Res(u,v)$ in the Graph"
    axs[4].set_title(title, fontsize=15)
    axs[4].set_xlabel("$Res(u,v)$", fontsize=14)
    axs[4].set_ylabel("Number of edges", fontsize=12)
    
    
    ######
    ## hist node ER
    bins = int(np.sqrt(G.number_of_nodes()))
    axs[5].hist(nodes_total_er,alpha=0.8,bins=bins, color='black')
    axs[5].set_title("$Res_G(u)$", fontsize=15)
    axs[5].set_xlabel("$Res_G(u)$", fontsize=14)
    axs[5].set_ylabel("Number of nodes", fontsize=12)
    
    
    
    ######
    ## hist diam ER
    axs[6].hist(nodes_betwn,alpha=1,bins=bins, color='black')
    title = "$Res_G(u, \mathcal{N})$"
    axs[6].set_title(title, fontsize=15)
    axs[6].set_xlabel("$\mathcal{R}_{diam}(u)$", fontsize=14)
    axs[6].set_ylabel("Number of nodes", fontsize=12)
    
    ######
    ## hist diam ER
    axs[7].hist(nodes_diams,alpha=1,bins=bins, color='black')
    axs[7].axvline(np.mean(nodes_diams), linewidth=2, color='r', label=f"$\overline{{\mathcal{{R}}_{{diam}}(V)}}$")
    title = "$\mathcal{R}_{diam}(u)$"
    axs[7].set_title(title, fontsize=15)
    axs[7].set_xlabel("$\mathcal{R}_{diam}(u)$", fontsize=14)
    axs[7].set_ylabel("Number of nodes", fontsize=12)
    axs[7].legend(fontsize=13)    
    
    #f.suptitle("Graph Metrics")
    
    f.tight_layout()
    plt.show()


def graph_effective_resistance_metrics(G, pos=None, node_idx=None, filtered=False):
    
    original_res = ermet.effective_resistance_matrix(G)
    evl, evc = spec.find_evecs(nx.laplacian_matrix(G))
    evl = np.real(evl)
    evc = np.real(evc)
    
    if node_idx:
        f, axs = plt.subplots(3,3, figsize=(14,12))
    else:
        f, axs = plt.subplots(2,3, figsize=(16,10))
    
    # Draw the graph
    pos = nx.kamada_kawai_layout(G) if pos is None else pos #spring_layout(G)
    options = {
        "node_color": np.array(evc[:,1]).ravel()+20,
        "edge_color": 'grey',
        "width": 1,
        "node_size":np.array([itm[1] for itm in dict(G.degree).items()])*10,
        "edge_cmap": plt.cm.seismic,
        "cmap": plt.cm.seismic,
        "with_labels": False,
    }
    axs[0][0].set_title('Graph (color=$\lambda_2$, size=$D$)', fontsize=20)
    nx.draw(G, pos, **options, ax=axs[0][0])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.seismic, 
                            norm=plt.Normalize(vmin=np.min(evc[:,1]), vmax=np.max(evc[:,1])))
    plt.colorbar(sm, ax=axs[0][0])
    
    axs[0][0].set_axis_on()
    total = ermet.total_effective_resistance(original_res)
    diam = ermet.resistance_diameter(original_res)
    avg_diam = ermet.avg_node_max_distance(original_res)
    title = f"Graph Metrics: $Res_G$: {total:.2e}\n"
    title += f"$\mathcal{{R}}_{{diam}}$: {diam:.2e}"
    title += f" - $\overline{{\mathcal{{R}}_{{diam}}(V)}}$: {avg_diam:.2e}"
    axs[0][0].set_xlabel(title, fontsize=13)
    
    ######
    ## Node total er as color node over the graph
    f_title = '$Res_G(u)$' if not filtered else '$Res_G(u, \mathcal{N})$'
    axs[0][1].set_title(f_title, fontsize=20)
    nodes_total_er = ermet.node_total_er(original_res, G=G, filtered=filtered)
    options['node_size'] = 100
    options['node_color'] = nodes_total_er
    options["cmap"] = plt.cm.viridis
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                            norm=plt.Normalize(vmin=np.min(nodes_total_er), vmax=np.max(nodes_total_er)))
    plt.colorbar(sm, ax=axs[0][1])
    axs[0][1].set_xlabel('$Res_G(u)$', fontsize=20)
    nx.draw(G, pos, **options, ax=axs[0][1])
    
    ######
    ## Node diameters as color node over the graph
    axs[0][2].set_title('$\mathcal{R}_{diam}(u)$', fontsize=20)
    nodes_diams = ermet.node_diameters(original_res)
    options['node_color'] = nodes_diams
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                            norm=plt.Normalize(vmin=np.min(nodes_diams), vmax=np.max(nodes_diams)))
    plt.colorbar(sm, ax=axs[0][2])
    nx.draw(G, pos, **options, ax=axs[0][2])
    
    ######
    ## Tots ER
    bins = int(np.sqrt(G.number_of_edges()))
    axs[1][0].hist(original_res[np.triu_indices(np.sum(G.number_of_nodes()),k=1)],alpha=1,bins=bins, color='black')
    title = "All $Res(u,v)$ in the Graph"
    axs[1][0].set_title(title, fontsize=15)
    axs[1][0].set_xlabel("$Res(u,v)$", fontsize=14)
    axs[1][0].set_ylabel("Number of edges", fontsize=12)
    
    
    ######
    ## hist node ER
    bins = int(np.sqrt(G.number_of_nodes()))
    axs[1][1].hist(nodes_total_er,alpha=0.8,bins=bins, color='black')
    axs[1][1].set_title("$Res_G(u)$ for each node", fontsize=15)
    axs[1][1].set_xlabel("$Res_G(u)$", fontsize=14)
    axs[1][1].set_ylabel("Number of nodes", fontsize=12)
    
    
    
    ######
    ## hist diam ER
    axs[1][2].hist(nodes_diams,alpha=1,bins=bins, color='black')
    axs[1][2].axvline(np.mean(nodes_diams), linewidth=2, color='r', label=f"$\overline{{\mathcal{{R}}_{{diam}}(V)}}$")
    title = "$\mathcal{R}_{diam}(u)$ for each node"
    axs[1][2].set_title(title, fontsize=15)
    axs[1][2].set_xlabel("$\mathcal{R}_{diam}(u)$", fontsize=14)
    axs[1][2].set_ylabel("Number of nodes", fontsize=12)
    axs[1][2].legend(fontsize=13)
    
    if node_idx:
        ######
        ## Distances in graph
        nx.draw_networkx_nodes(G, pos=pos, 
                           node_size=40, 
                           nodelist=list(G.nodes),
                           node_color=original_res[node_idx,:],
                           cmap=plt.cm.viridis,
                           edgecolors=None,
                           label='CT distance', ax=axs[2][0])
        
        nx.draw_networkx_nodes(G, pos=pos, 
                               node_size=100,
                               nodelist=[node_idx], 
                               node_color='red',
                               edgecolors='black', 
                               node_shape = 'v',
                               label='Source node', ax=axs[2][0])
        
        nx.draw_networkx_edges(G, pos=pos, edge_color='grey', ax=axs[2][0])
        axs[2][0].legend()
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                            norm=plt.Normalize(vmin=np.min(original_res[node_idx,:]), vmax=np.max(original_res[node_idx,:])))
        plt.colorbar(sm, ax=axs[2][0])
        axs[2][0].set_axis_off()
        
        ######
        ## node histogram distances
        axs[1][0].hist(original_res[node_idx,:],alpha=1,bins=int(np.sqrt(G.number_of_edges())), label=f"node {node_idx}")
        axs[1][0].legend()
        axs[2][1].hist(original_res[node_idx,:],alpha=1,bins=bins, color='black')
        axs[2][1].set_title(f"$Res(u,v)$ for u={node_idx}", fontsize=15)
        axs[2][1].set_xlabel("$Res(u,v)$", fontsize=14)
        axs[2][1].set_ylabel("Number of nodes", fontsize=12)
        

        ######
        ## node metrics
        total_node = ermet.node_total_er(original_res)[node_idx]
        diam_node = ermet.node_diameters(original_res)[node_idx]
        text =  f"Graph Metrics:\n$Res_G$: {total:.2e}\n"
        text += f"$\mathcal{{R}}_{{diam}}$: {diam:.2e}\n"
        text += f"$\overline{{\mathcal{{R}}_{{diam}}(V)}}$: {avg_diam:.2e}"
        text += "\n\n\n"
        text += f"Node Metrics:\n$Res_G(u)$: {total_node:.2e}\n"
        text += f"$\mathcal{{R}}_{{diam}}(u)$: {diam_node:.2e}"
        axs[2, 2].set_axis_off()
        axs[2, 2].text(0, 0, text, fontsize=13)
        
    
    
    #f.suptitle("Graph Metrics")
    
    f.tight_layout()
    plt.show()


def compare_graphs(graph_list, graph_names=None, pos=None, edge_highlight=False, node_size=5, save=False):
    
    ori_G = graph_list[0]
    original_res = ermet.effective_resistance_matrix(ori_G)
    ori_nodes_total_er = ermet.node_total_er(original_res)
    ori_nodes_betwn = ermet.node_total_er(original_res, G=ori_G, filtered=True)
    ori_nodes_diams = ermet.node_diameters(original_res)
    #evl, evc = find_evecs(nx.laplacian_matrix(G))
    #evl = np.real(evl)
    #evc = np.real(evc)
    
    """MOSAIC=''
    for i in range(len(graph_list)):
        j, k, l, m = chr(i*4+97), chr(i*4+98), chr(i*4+99), chr(i*4+100)
        MOSAIC += f"{j}{k}{l}{m}\n"
    f,axs_dict = plt.subplot_mosaic(MOSAIC,
                                    figsize=(14,4*len(graph_list)),
                                    gridspec_kw={'width_ratios': [.5, 1, 1, 1]})"""
    f, axs = plt.subplots(len(graph_list),4, 
                          figsize=(14,4*len(graph_list)),
                         gridspec_kw={'width_ratios': [.4, 1, 1, 1]})
    
    # Draw the graph
    pos = nx.kamada_kawai_layout(ori_G) if pos is None else pos #spring_layout(G)
    options = {
        "edge_color": 'grey',
        "width": 0.8,
        "node_size":np.array([itm[1] for itm in dict(ori_G.degree).items()])*node_size,
        "edge_cmap": plt.cm.seismic,
        "cmap": plt.cm.seismic,
        "with_labels": False,
    }
    
    axs[0][1].set_title("$Res_G(u)$", fontsize=16)
    axs[0][2].set_title('\n'+'\mathsf{B_R}(u)$', fontsize=16)
    axs[0][3].set_title('$\mathcal{R}_{diam}(u)$', fontsize=16)
    

    for idx, G in enumerate(graph_list):
        
        resistance = ermet.effective_resistance_matrix(G)
        
        total = ermet.total_effective_resistance(resistance)
        diam = ermet.resistance_diameter(resistance)
        avg_diam = ermet.avg_node_max_distance(resistance)
        
        suptitle = f"\n\n$Res_G$: {total:.2e}\n"
        suptitle += f"$\mathcal{{R}}_{{diam}}$: {diam:.2e}\n"
        suptitle += f"$\overline{{\mathcal{{R}}_{{diam}}(V)}}$: {avg_diam:.2e}"
        axs[idx][0].axis('off')
        axs[idx][0].text(1, 1, f'{graph_names[idx]}', ha='right', va='top', fontsize=14)
        axs[idx][0].text(1, 1, suptitle, ha='right', va='top', fontsize=12)

        ######
        ## Node total er as color node over the graph
        nodes_total_er = ermet.node_total_er(resistance)
        
        #f_title = '$Res_G(u)$'
        #axs[idx][1].set_title(f_title, fontsize=14)
        options['node_color'] = nodes_total_er
        options["cmap"] = plt.cm.viridis
        nx.draw(G, pos, **options, ax=axs[idx][1],
                vmin =ori_nodes_total_er.min() , vmax=ori_nodes_total_er.max())        

        ######
        ## Node diameters as color node over the graph
        nodes_betwn = ermet.node_total_er(resistance, G=G, filtered=True)
        
        #axs[idx*4+2].set_title('\n'+'$Res_G(u, \mathcal{N})$', fontsize=14)
        options['node_color'] = nodes_betwn
        nx.draw(G, pos, **options, ax=axs[idx][2],
                vmin =ori_nodes_betwn.min() , vmax=ori_nodes_betwn.max())

        ######
        ## Node diameters as color node over the graph
        nodes_diams = ermet.node_diameters(resistance)
        #axs[idx*4+3].set_title('$\mathcal{R}_{diam}(u)$', fontsize=14)
        options['node_color'] = nodes_diams
        nx.draw(G, pos, **options, ax=axs[idx][3],
                vmin =ori_nodes_diams.min() , vmax=ori_nodes_diams.max())
        
        
        if edge_highlight:
            G_diff_edges = list(ori_G.edges() ^ G.edges())
            nx.draw_networkx_edges(G, edgelist = G_diff_edges,
                                   pos=pos, edge_color='orange', ax = axs[idx][1])
            nx.draw_networkx_edges(G, edgelist = G_diff_edges,
                                   pos=pos, edge_color='orange', ax = axs[idx][2])
            nx.draw_networkx_edges(G, edgelist = G_diff_edges,
                                   pos=pos, edge_color='orange', ax = axs[idx][3])
            
    
    ### COLORBARS
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                norm=plt.Normalize(vmin=np.min(ori_nodes_total_er), 
                                                   vmax=np.max(ori_nodes_total_er)))
    cbar = plt.colorbar(sm, ax=axs[:,1], shrink=0.5)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_linewidth(0)
        
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                norm=plt.Normalize(vmin=np.min(ori_nodes_betwn), 
                                                   vmax=np.max(ori_nodes_betwn)))
    cbar = plt.colorbar(sm, ax=axs[:, 2], shrink=0.5)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_linewidth(0)
            
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                norm=plt.Normalize(vmin=np.min(ori_nodes_diams), 
                                                   vmax=np.max(ori_nodes_diams)))
    cbar = f.colorbar(sm, ax=axs[:, 3], format='%.1f', shrink=0.5) #axs[idx*4+3]
    cbar.ax.tick_params(labelsize=12, labelleft=True, labelright=False, left=True, right=False)
    cbar.outline.set_linewidth(0)
    
    
    #plt.subplots_adjust(wspace=-0.1)
    #f.tight_layout()
    if save:
        f.savefig('nodemetricsexample.pdf', dpi=300, bbox_inches='tight')
    plt.show()