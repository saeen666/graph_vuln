import pathlib, glob, os, time, re, json
from tabulate import tabulate
from halo import HaloNotebook as Halo
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
import networkx as nx
import pygraphviz
import scipy.sparse as sp
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models.word2vec import Word2Vec
import src.data as data_util
from src.utils.functions.parse import tokenizer
import src.data as data_util
import src.process as process
from torch_geometric.data import Data
from torch_geometric.utils import convert

class NodesEmbedding:
    def __init__(self, nodes_dim: int, w2v_keyed_vectors: Word2VecKeyedVectors):
        self.w2v_keyed_vectors = w2v_keyed_vectors
        self.kv_size = w2v_keyed_vectors.vector_size
        self.nodes_dim = nodes_dim

        assert self.nodes_dim >= 0

        # Buffer for embeddings with padding
        self.target = torch.zeros(self.nodes_dim, self.kv_size).float()

    def __call__(self, nodes):
        embedded_nodes = self.embed_nodes(nodes)

        nodes_tensor = torch.from_numpy(embedded_nodes).float()

        self.target[:nodes_tensor.size(0), :] = nodes_tensor

        return self.target

    def embed_nodes(self, G):
        embeddings = []
        g_nodes = {}
        for (n,d) in G.nodes(data=True):
            # Get node's code
            node_code = d
            # Tokenize the code
            tokenized_code = tokenizer("".join(d.values()))
            if not tokenized_code:
                # print(f"Dropped node {node}: tokenized code is empty.")
                msg = f"Empty TOKENIZED from node CODE {node_code}"
                print(msg)
            # Get each token's learned embedding vector
            vectorized_code = np.array(self.get_vectors(tokenized_code))
            # The node's source embedding is the average of it's embedded tokens
            source_embedding = np.mean(vectorized_code, 0)
            # The node representation is the concatenation of label and source embeddings
            #embedding = np.concatenate((np.array([node.type]), source_embedding), axis=0)
            embeddings.append(source_embedding)
        # print(node.label, node.properties.properties.get("METHOD_FULL_NAME"))
        
        return np.array(embeddings)

    # fromTokenToVectors
    def get_vectors(self, tokenized_code):
        vectors = []
        for token in tokenized_code:
            if token in self.w2v_keyed_vectors.key_to_index:
                vectors.append(self.w2v_keyed_vectors[token])
            else:
                # print(node.label, token, node.get_code(), tokenized_code)
                vectors.append(np.zeros(self.kv_size))
        return vectors



def nodes_to_input(G, target, nodes_dim, keyed_vectors):
    nodes_embedding = NodesEmbedding(nodes_dim, keyed_vectors)
    edge_index, edge_attr = convert.from_scipy_sparse_matrix(nx.adjacency_matrix(G))
    label = torch.tensor([target]).float()

    return Data(x=nodes_embedding(G), edge_index=edge_index, edge_attr=edge_attr ,y=label)


#change these arguments where word2vec model is defined
#change the name of word2vec model

'''word2vec_args={
        "vector_size": 200,
        "alpha": 0.01,
        "window": 5,
        "min_count": 3,
        "sample": 1e-05,
        "workers": 8,
        "sg": 1,
        "hs": 0,
        "negative": 5,
        "epochs":5,
    }'''



check_empty  ='digraph .*\{ *\n*\}' #to check if generated CFG is empty or not
pd_list = []
dict_e = {}
path_code = 'dataset_mini' #I have changed the name
project_path = glob.glob(path_code+"/*_cfg")
for i in tqdm(project_path):
    path = pathlib.PurePath(i)
    project_name = path.name 
    project = glob.glob(i+'/*')
    dict_e[i+' total'] = len(project)
    dict_e[i+' removed'] = 0
    for path_src in tqdm(project, 'total files'):
        #print(path_src)
        new_dict   = dict()
        src_file   = str(Path(path_src+'/'+pathlib.PurePath(path_src).name+'.c').resolve())
        cfg_folder = str(Path(path_src+'/'+'cfg/').resolve())
        index      = int(pathlib.PurePath(src_file).name.replace(".c",""))
        target     = project_name.split('_')[0]
        src_file   =  str(Path(os.path.join('', *[path_code,target,str(index)+'_c.c'])).resolve())
        #print('{}- {}- {}- {}- {}'.format(path_src, cfg_folder,index,target,src_file))
        with open(src_file, 'r') as f:
            src_code = f.read()
        dot_arr = []
        for file in glob.glob(cfg_folder+"/*"):
            with open(file,'r') as f:
                dot_arr.append(f.read())
        dot_arr = [x for x in dot_arr if not re.search(check_empty, x)] # removes empty graphs
        if (len(dot_arr) == 1):                                         #if graph is not empty and is connected
            is_connected = True
            #G = nx.Graph(nx.drawing.nx_pydot.read_dot(Path(cfg_folder).joinpath("0-cfg.dot")))
            try:
                with open(Path(cfg_folder).joinpath("0-cfg.dot")) as f:
                    dotFormat = f.read()
                new_str = dotFormat.replace('\\"', '')                       #To catch escape characters
                new_str = "\n".join([f_str.strip() for f_str in dotFormat.split('\n')])
                G = nx.drawing.nx_agraph.from_agraph(pygraphviz.AGraph(new_str)); #convert graph into Networkx object
                new_dict['index']         = index
                new_dict['project']       = project_name
                new_dict['func_code']     = src_code
                new_dict['graph']         = G
                new_dict['is_connected']  = is_connected
                new_dict['dot_string']    = new_str
                if target=='positive':
                    new_dict['target']        = 1
                elif target =='negative':
                    new_dict['target']        = 0
                pd_list.append(new_dict)
            except ValueError as e:
                dict_e[i+' removed'] += 1
                pass
            
        dict_e[i+' used'] = dict_e[i+' total'] - dict_e[i+' removed']
        

print(json.dumps(dict_e, indent=4))
data = pd.DataFrame(pd_list)
data = data[['target', 'project', 'graph','func_code','index']]
data = data.rename(columns={'func_code': 'func'})
print('tokenizing code')
tokens_dataset = data_util.tokenize(data)
spinner = Halo(text='Training word2vec on tokens code', spinner='dots')
w2vmodel = Word2Vec(sentences=tokens_dataset.tokens, vector_size=200, 
                        window=5, min_count=1, workers=4, epochs=10)
                        
spinner.stop()
print("Saving w2vmodel.")
w2vmodel.save("word2vec.model")

print("No of samples in dataset: {} ".format(len(data)))
print('\n'*3,"*"*40,'\n')

node_size_group = data.apply(lambda g: nx.number_of_nodes(g.graph),axis=1).describe()[['min', 'max','mean','std']]

print(tabulate(node_size_group.to_frame(),
               tablefmt="grid", stralign='left', numalign='left',
               headers=['Node stats']))


print('\n'*3)

edge_size_group = data.apply(lambda g: nx.number_of_edges(g.graph),axis=1).describe()[['min', 'max','mean','std']]
print(tabulate(edge_size_group.to_frame(),
                   tablefmt="grid", stralign='left', numalign='left',
                   headers=['Edge stats']))

print('\n\n')
print("1 = Vulnerable, 0 = Not Vulnerable")
print('\n')
for name, group in data.groupby('target'):
    
    edge_size_group = group.apply(lambda g: nx.number_of_edges(g.graph),axis=1).describe()[['min', 'max','mean','std']]
    node_size_group = group.apply(lambda g: nx.number_of_nodes(g.graph),axis=1).describe()[['min', 'max','mean','std']]
    
    print(tabulate(node_size_group.to_frame(),
               tablefmt="grid", stralign='left', numalign='left',
               headers=['Class {} Node stats'.format(name)]))
    
    print('\n'*3)
    print(tabulate(edge_size_group.to_frame(),
                   tablefmt="grid", stralign='left', numalign='left',
                   headers=['Class {} Edge stats'.format(name)]))
    print('\n'*3)
    


data["input"] = data.apply(lambda row: nodes_to_input(row.graph, row.target, nx.number_of_nodes(row.graph),
                                                                                    w2vmodel.wv), axis=1)
print('Writing to file/pandas')
pd.to_pickle(data[['input','target']], 'new_data_mini')