import networkx as nx
import numpy as np
import math
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import os
import pickle as pkl

def compute_ricci(args, data_path):
    G = load_graph(args.dataset, args.use_feats, data_path)
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    G_orc = orc.G.copy()
    summary_stats(G_orc)

def summary_stats(G):
    print("Sanity check of first 5 edges:")
    for n1,n2 in list(G.edges())[:5]:
        print("Ollivier-Ricci curvature of edge (%s,%s) is %f" % (n1 ,n2, G[n1][n2]["ricciCurvature"]))
    ricci_curvtures = nx.get_edge_attributes(G, "ricciCurvature").values()
    print("Mean Ricci Curvatures: {}".format(np.mean(ricci_curvtures)))

def load_graph(dataset_str, use_feats, data_path):
    if dataset in ['cora', 'pubmed']:
        with open(os.path.join(data_path, "ind.{}.graph".format(dataset_str)), 'rb') as f:
            if sys.version_info > (3, 0):
                graph = pkl.load(f, encoding='latin1')
            else:
                graph = pkl.load(f)
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        G = nx.from_numpy_matrix(adj)
    elif dataset == 'disease_lp':
        pass
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    return G
