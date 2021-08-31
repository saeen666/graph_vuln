# graph_vuln
Code related to graph generation from source code and identifying vulnerability in them using GNN


### Make sure following packages are installed
- Python
    - pandas
    - numpy
    - tqdm
    - networkx
    - pygraphviz
    - scipy
    - sklearn
    - spektral
    - tensorflow
    - tokenizer_c (tokenizer code)
- System
    - Joern
    - graphviz

## Use the CFG_code_ultimate.ipynb to run all of the following functions
- I have commented the source code to make it more clear
- Please place the tokenizer, dataset JSON file, and notebook file in the same directory
- Contact me if you find any bug or issue

 ## 1. Generate CFG using the following method
- make dir for each project i.e. in our case it is will be two folders one for FFmpeg and other one is for qemu defined by variable path_code
- define your path where joern is installed using variable joern_path
- read the dataset and Extract all instances of source code and group them based on their project
- iterate over each project group and save the source code in a .c file. The name of the file is the index where the source file is saved


### 1.2. Extract CFG from the .c files using Joern
- The Joern generate CFG from source code is two steps'
    1. Export the CPG file from the source code
    2. Parse the extracted CPG file in any supported representation: AST, CFG, PDG, CPG, etc.
- We have saved CPGs for the entire dataset and now when extracting any required graph representation we only have to run step 2
- The parallelization doesn't work that great with joern. To speed up the process run Joern on each separate group
- The Entire process takes around 2-3 Hours on sir's machine

## 2. Compiling CFGs into graph object and storing them in Pandas Dataframe

### 2.1. Extracting unique operation keywords in the dataset for node embedding
- I have extracted all the unique C-tokens that are used in the dataset
- I also have extracted all the unique operation words used in the dataset
- As you can see I have hardcoded the unique tokens and operations the reasons are:
  - There are actually 84 unique tokens in the tokenizer dictionary but in our dataset, there are only 64 unique tokens. That's why I discarded the other tokens to reduce the dimension of node embedding. You can get all the available tokens in the tokenizer by running the code tokenizer_c.tokens
  - The reason I have also hardcoded the unique operations is that I didn't want to compute all the operators every time. You can get all the unique operation words by running the following function
- To encode the node features (source code) I have done the following:
  - I have tokenized the whole code available at the node and selected only unique tokens from it
  - I have gotten the operation word at each node can compare it with the type of operations I have. It is possible that a source code contains function calling code and it has no operation type so the operation type will be a zero vector
- I have concatenated one-hot encoding of both representation
- I have stored the embeddings in a pickle file and I will integrate them into the graphs in the next step

## 2.2 Merging node features into graph
- I have merged the one hot encoding embeddings with the graph in following code

## 3 Training on both projects using GIN
- Please configure the model's parameters before running
- You can provide training and testing set ratio
- If someone can also confirm the code of GIN is authentic ? I am using from spektral
