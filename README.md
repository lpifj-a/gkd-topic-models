# Generalized Knowledge Distillation for Topic Models

This code is a framework for general knowledge distillation for neural topic models based on the code of [Scholar+BAT](https://github.com/ahoho/kd-topic-models) and [CRCD](https://github.com/Lechatelia/CRCD)

# Quick start
[!Note]
The file paths have been converted from absolute paths to relative paths for publication, which may cause errors. In this case, please rewrite the paths properly.

## 1. Create conda environment
```
conda env create -f gkd_topic_models.yml
```
## 2. Preprocess the data

### Target data
#### 20 Newsgroups
Go to /data/20ng/ and run the .py files in the order of 1 to 6.
```
python 1_convert_prodlda_to_txt_py27.py
python 2_convert_txt_to_scholar_format_py3.py
python 3_replicate_and_align_raw_data.py
python 4_create_dev_sets.py
python 5_create_aligned_dev_set.py
python 6_create_raw_text_file.py
```

#### IMDb
Download the data
```
python data/imdb/download_imdb.py
```
Main preprocessing script
```
python data/imdb/preprocess_data.py ./data/imdb/train.jsonlist ./data/imdb/processed --vocab-size 5000 --test ./data/imdb/test.jsonlist --label rating
```
Create a dev split from the train data
```
cd data/imdb
python data/imdb/create_dev_split.py
```

### BBC
Download the preprocessed datasets from [this web page](http://mlg.ucd.ie/datasets/bbc.html) and place them in `data/bbc-processed/`. Then convert them to a format that can be trained by scholar
```
python data/bbc-processed/convert_scholar_format.py
```

### Source data

#### Wiki
Download the data
```
python data/wiki20200501/download_wiki.py
```
Main preprocessing script
```
python scholar/preprocess_data.py ./data/wiki20200501/train.jsonlist ./data/wiki20200501/processed --vocab-size 50000
```
Create a dev split from the train data
```
python data/wiki20200501/create_dev_split.py
```

#### IMDb
Download the data
```
python data/imdb/download_imdb.py
```
Main preprocessing script (No preprocessing on vocabulary counts)
```
python scholar/preprocess_data.py ./data/imdb/train.jsonlist ./data/imdb/processed_source --test ./data/imdb/test.jsonlist --label rating
```
Create a dev split from the train data
```
cd data/imdb
python data/imdb/create_dev_split_source.py
```


## 3. Teacher model
We propose two ways of using a neural topic model or an LLM as a teacher model for knowledge distillation.
### 3.1.  Run the teacher toipic model
Pre-training the parameters of the inference network of the neural topic model using the source dataset (ex. Wiki)

```
python scholar/run_scholar.py ./data/wiki20200501/processed -k 500 --emb-dim 500  --epochs 500 --batch-size 5000 --background-embeddings --device 0 -l 0.002 --alpha 0.5 --eta-bn-anneal-step-const 0.25 -o ./outputs/wiki/wiki_topic_500_emb_dim_500  --save-for-each-epoch 10 
```

Fine-tuning the neural topic model with the target datset (ex. IMDb) using the obtained parameters.

```
python scholar/init_embeddings.py ./data/imdb/processed-dev/train.vocab.json  --teacher-vocab ./data/wiki20200501/processed/train.vocab.json  --model-file outputs/wiki/wiki_topic_500_emb_dim_500/torch_model_epoch_100.pt --emb-dim 500 -o ./scholar/outputs/imdb/teacher_weight/wiki_topic_500_emb_500_epoch_100
```

```
python scholar/multiple_run_scholar_crcd.py $(cat args/imdb/scholar_wiki.txt)
```

### 3.2. Create teacher LLM embeddings 
LLM embedding is created using [LangChain](https://python.langchain.com/docs/integrations/text_embedding/llamacpp). This study uses [Yarn Llama 2 13B 128K - GGUF](https://huggingface.co/TheBloke/Yarn-Llama-2-13B-128K-GGUF), a lightweight version of [Yarn Llama 2 13B 128K](https://huggingface.co/NousResearch/Yarn-Llama-2-13b-128k), as the LLM. The LLM to be used is placed in `embeddings/llama_cpp/` (ex. `embeddings/llama_cpp/20ng/models/Yarn-Llama-2-13B-128K-GGUF.gguf`).
#### Enable GPU
```
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```
(ref. https://python.langchain.com/docs/integrations/llms/llamacpp)

#### Create LLM embeddings
20 Newsgroups
```
CUDA_VISIBLE_DEVICES=1,2,3,4,5 python embeddings/llama_cpp/20ng/create_embeddings.py 
```

## 4. Knowledge Distillation 

### Teacher: Topic Model
- Target data: IMDb, Source data: Wiki
```
python scholar/multiple_run_scholar_crcd.py $(cat scholar/args/imdb/wiki_ResKD_FeaKD_RCD_opt.txt)
```
- Target data: 20NG, Source data: IMDb
```
python scholar/multiple_run_scholar_crcd.py $(cat scholar/args/20ng/imdb_ResKD_FeaKD_RCD_opt.txt)
```

### Teacher: LLM
- Target data: 20NG
```
python scholar/multiple_run_scholar_crcd.py $(cat scholar/args/20ng/scholar_FeaKD_MSE_RCD_yarn_llama_cpp.txt)
```

# Hyperparameter tuning
Use `run_scholar_crcd_optuna.py` to search for hyperparameters. For example, a generalized knowledge distillation hyperparameter search using LLM as a teacher can be run as follows:
```
python scholar/run_scholar_crcd_optuna.py $(cat scholar/args/20ng/optuna/feakd_mse_rcd_llama_optuna.txt)
```