# graph_vuln
Code related to graph generation from source code and identifying vulnerability in them using GNNs


## Make sure following packages are installed
- Python
    - pandas
    - numpy
    - tqdm
    - networkx
    - ply
    - pygraphviz
    - scipy
    - sklearn
    - spektral
    - tensorflow
    
- System
    - Joern
    - graphviz

### Please make sure that these conditions are fulfilled:
- Please place the tokenizer, dataset JSON file, and notebook file in the same directory
- *You can download the data in JSON format from [here](https://drive.google.com/open?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF)**
- Contact me if you find any bug or issue

# Use the jupyter notebook "CFG_code_complete.ipynb" to run all of the following functions:


# Generating CFG from devign dataset and training it on GIN

 ## 1.1 Generate CFG using the following method
- make dir for each project, in our case it is will be two folders one for FFmpeg and another for qemu 
- the parent folder of these newly build directories must be defined through variable: path_code

- The code will read the dataset and extract all instances of source code and group them based on their project
- It will iterate over each project and save the source code in a .c file. The name of the file is the index where the source file is saved


## 1.2 Extract CFG from the .c files using Joern
- Define path where joern_cli is installed by using variable joern_path
- The Joern generate CFG from source code is two steps'
    1. Export the CPG file from the source code
    2. Parse the extracted CPG file in any supported representation: AST, CFG, PDG, CPG, etc.
- The code saves CPGs for the entire dataset. Now the graph representation can be extracted without running step 1 again
- Parallelization doesn't increase efficiency while using joern. To speed up the process run Joern on each separate group/folder
- The Entire process takes around 2-3 Hours on machine with 66gb ram, 16 core i9 processor 

## 1.3 Compiling CFGs into graph objects and storing them in Pandas Dataframe

### 1.3.1 Extracting unique operation keywords in the dataset for node embedding
- It extracts all the unique C-tokens that are used in the dataset
- It also has extracted all the unique operation used in the dataset
- I have hardcoded the unique tokens and operations the reasons are:
  - There are 84 unique tokens in the tokenizer dictionary but there are only 64 unique tokens in our dataset. That's why I discarded the other tokens to reduce the dimension of node embeddings. You can get all the available tokens in the tokenizer by running the code/command: tokenizer_c.tokens
  - The reason I have also hardcoded the unique operations is that I didn't want to compute all the operators every time.
- To encode the node features (source code) I have done the following:
  - It tokenizes the whole code available at the node and selected unique tokens from it
  - It extracts the operation type and constructs one hot encoding from them. Sometimes there might be an empty vector. Like a source code may contain a function calling code and it has no operation type so the operation type will be a zero vector. There should be a way to identify function calls in node-level information, we might need to work on this.
- It concatenates one-hot encoding of both representation obtained from tokenized code and operation type
- It stores the embeddings in a pickle file

## 1.4 Merging node features into a graph
- I have merged the one-hot encoding embeddings with the graph in the this part code

## 1.5 Training on both projects using GIN
- Please configure the model's parameters before running
- You can provide training and testing set ratio
- If someone can also confirm the code of GIN is authentic? I am using the implementation from spektral


# Generating word2vec training on big_vul dataset
- use all_three_tasks.ipynb for the tasks. Please read the above description before running this code. 
