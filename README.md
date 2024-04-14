# ntu-msds-ai6103
Repository for AI6103 Deep Learning project.

## Collaboration Structure

Let's observe the following  collaboration structure to en$$sure that the code repository remains friendly, organised and transparent to all collaborators.

- Do create a branch with your name and add your contributions there. Once you have finished, please make a pull request before merging into the main branch
  - As far as possible, please observe the usual git protocols and workflows while balancing with the ease of contribution
  - No approvers are needed to complete the pull requests, so feel free to merge your own PR into main at your own risk. 
- Files should be organised into their respective subdirectory. For instance, notebooks should be under [notebooks](notebooks)
- Do comment on the code if the code is meant to be shared with others for illustration/descriptive purposes.
- If applicable, please annotate your name at the top of each file so others  can identify who is responsible for which artifacts.


## Project Directory

Files are organised based on the structure below. Feel free to extend the subdirectories (E.g. based on their functionailities/categories etc.) within these folders. Here's a screenshot of how the preliminary project structure looks like. 

![directory screenshot](assets/img/directory.png)

- [Notebooks](notebooks): for all notebooks. 
- [Source Scripts](src): for source scripts, either as standalone  Python files, modules or embedded within a Jupyter Notebook.
- [Dependencies](requirements.txt): python dependencies
- [Data](data): data artifacts, includes raw, staging or production data
- [Assets](assets): for other relevant assets such as images, model  files etc.


## Installation & Usage

### Environment
Set up environments
``` shell
# linux
python3 -m venv .venv # create venv

source .venv/bin/activate # activate venv

source deactivate # deactivate venv

# windows
python -m venv .venv # create venv

.venv/Scripts/Activate # activate venv

deactivate # deactivate venv
```

Update dependencies using pip
``` shell
# update dependencies
pip freeze > requirements.txt

# install dependencies
pip install -r requirements.txt
```

### Complete Execution
```
# step 1
python build_graph.py --gen_sem --gen_seq --gen_syn --dataset R8  --corenlp ../stanford-corenlp-4.5.6 

# step 2, based on step 1's execution, look for the log timestamp as the input for the --run_id flag below. This will point the training script to use the graph adjacencies from the previous graph execution.
 
$ python train.py --do_train --do_valid --do_test --dataset 20ng --run_id <timestamp>

```

### Building graph
```
# cd into directory
$ cd Users/LSUN011/ntu-msds-ai6103/src/

# building multi graphs 
$ python build_graph.py --gen_sem --gen_seq --gen_syn --dataset R8  --corenlp ../stanford-corenlp-4.5.6

$ python build_graph.py --gen_sem --dataset 20ng --corenlp ../stanford-corenlp-4.5.6

# building only semantic graph with default epochs
$ python build_graph.py --gen_sem --dataset 20ng --corenlp ../stanford-corenlp-4.5.6 --epochs 1 
```

### Training

Training using all 3 graphs
```
# test training scrith
$ python train.py --do_train --do_valid --do_test --dataset R8 --epochs 10

$ python train.py --do_train --do_valid --do_test --dataset 20ng
```

Training using 1 or 2 graphs  

`train.py` allows dynamic training using only selected graphs (1 or 2 graphs). This relies on the argument `args.run_id`. A subdirectory under `src/saved_graphs/run_<id>` has to be prepared containing the selected graphs. For instance, if only sequential graph adjacency is available, the training script will only use the sequential graph adjacency for training. If syntactic and semantic graphs are available, it will use both graph. if all 3 graphs are available, all 3 graphs will be used. Note that the `graph_config.json` file has to be available within the same directory as well.

```
using src/saved_graphs/run_2024-04-09_20-14-25 as the target directory containing the graphs.
$ python -m train --do_train --do_valid --do_test  --dataset R8 --epochs 1 --run_id 2024-04-09_20-14-25
```

### Logging
Dynamic logging is integrated to create log files under `logs`. 
- `build_graph.py` will output log files into `logs/graph_<timestamp>` accordingly. Note that timestamp is the execution timestamp dynamically created.
- `train.py` will output log files into `logs/model_<timestamp>`. Note that timestamp is the execution timestamp dynamically created. The timestamp for training logs will also match the timestamp for saved_models folder. For more details on persisting artifacts, head to the next section.

### Persisting Artifacts
Dynamic persisting of artifacts are also enabled by default when building graphs and training the model. The output for persisting graph and model artifacts are found under `src/saved_graphs` and `src/saved_models `respectively. 

To point `train.py` to retrieve graph adjacencies from a specific run, use the flag --run_id. For instance: `python -m train --do_train --do_valid --do_test  --dataset R8 --epochs 1 --run_id 2024-04-09_20-14-25`.
