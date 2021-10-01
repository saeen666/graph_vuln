import pathlib, glob, time, subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import numpy as np


graph_type = 'cfg'
path_code = pathlib.Path('dataset1/')                       
joern_path = '/home/anwar/software_vulnerability/joern-cli/'

# iterate over each group/project
for folder in tqdm(glob.glob('{}/*'.format(path_code))):
    if graph_type not in folder:
        print(folder)
        path_group = pathlib.Path('{}_{}'.format(folder,graph_type))         #make folder for the dataset
        path_group.mkdir(parents=True, exist_ok=True)
    
        start_time = time.time()

        project = glob.glob(folder+'/*')                  #get all .c files we just generated

        for path_src in tqdm(project):
            src_file      = str(Path(path_src).resolve())
            out_path      = str(Path(str(path_group)+'/'+path_src.split('/')[-1].replace('_c.c','')).resolve())+"/"
            file_dir      = pathlib.Path(str(Path(str(path_group)+'/'+path_src.split('/')[-1].replace('_c.c','')).resolve())+"/")         #make folder for the dataset
            file_dir.mkdir(parents=True, exist_ok=True)

            cpg_path_cmd  = [joern_path+"./joern-parse",src_file,"--out",out_path+src_file.split("/")[-1].replace("_c.c","")+".cpg"]
            graph_out_cmd = [joern_path+"/joern-export",out_path+src_file.split("/")[-1].replace("_c.c","")+'.cpg',"--repr",graph_type,'--out',out_path+'/cfg/']
            cpg_check     = Path(out_path+src_file.split("/")[-1].replace("_c.c","")+".cpg").is_file()

            if cpg_check and not Path(out_path+'cfg/').is_dir():
                result = subprocess.call(graph_out_cmd,stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

            # check if CFG and both CPG are present if true nothing will be generated
            elif cpg_check and Path(out_path+'cfg/').is_dir():
                pass

            else:
            #if both of them all false compute CPG and generate CFG 
                result = subprocess.call(cpg_path_cmd,stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

                result1 = subprocess.call(graph_out_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        print("--- %s miuntes ---" % ((time.time() - start_time)/60))
        