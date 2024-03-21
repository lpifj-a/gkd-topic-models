import os
import sys
import argparse

import gensim
import git
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

import file_handling as fh
from scholar import Scholar
from compute_npmi import compute_npmi_at_n_during_training
from run_scholar import compute_npmi_at_n_during_training, load_word_counts,load_scholar_model, get_topic_diversity, evaluate_perplexity

parser = argparse.ArgumentParser()
parser.add_argument("--restart", action='store_true')
args = parser.parse_args()


input_dir = "/home/watanabe/kd-topic-models/data/wiki_train_dev/processed-dev"
model_dir = "/home/watanabe/kd-topic-models/outputs/wiki_topic_500_emb_dim_500/"
model_list = os.listdir(model_dir)
print(model_list)

_, vocab, _, _ = load_word_counts(input_dir, "train")
dev_X, vocab, dev_row_selector, dev_ids = load_word_counts(input_dir, "dev", vocab=vocab)

embeddings = {}
embeddings["background"] = (None, True)
batch_size=5000


if args.restart:
    re_df = pd.read_csv("/home/watanabe/kd-topic-models/outputs/eval/eval.csv")
    print("computed:",re_df.columns[1:])
    loc = len(model_list) - len(re_df.columns[1:])
    print("computing",model_list[-loc:])
    
    for name in tqdm(model_list[-loc:]):
        model_path = model_dir + name
        model, _ = load_scholar_model(model_path,embeddings)
        model.eval()
        epoch = re.sub(r"\D", "", name)
        
        print("Computing Dev Perplexity")
        perplexity = 0.0
        perplexity = evaluate_perplexity(
            model,
            dev_X,
            None,
            None,
            None,
            None,
            None,
            batch_size,
            eta_bn_prop=0.0,
            )

        print("Computing Dev NPMI")
        npmi,_ = compute_npmi_at_n_during_training(
            model.get_weights(), ref_counts=dev_X.tocsc(), n=10, smoothing=0.,
        )

        print("Computing Topic diversity")
        topic_diversity = get_topic_diversity(model.get_weights(),25)


        print(f"epoch:{epoch}")
        print(f"perplexity:{perplexity:0.4f}")
        print(f"npmi:{npmi:0.4f}")
        print("topic diversity", topic_diversity)

        re_df[epoch] = [perplexity, npmi, topic_diversity]
        # メモリ解放
        del perplexity
    print(re_df)
    re_df.set_index("metric", inplace=True)
    re_df.to_csv("/home/watanabe/kd-topic-models/outputs/eval/eval.csv")


else:
    idx=["metric"]
    perplexity_list=["perplexity"]
    npmi_list=["npmi"]
    TD_list=["topic_diversity"]

    for name in tqdm(model_list):
        model_path = model_dir + name
        model, _ = load_scholar_model(model_path,embeddings)
        model.eval()
        epoch = re.sub(r"\D", "", name)
        
        print("Computing Dev Perplexity")
        perplexity = 0.0
        perplexity = evaluate_perplexity(
            model,
            dev_X,
            None,
            None,
            None,
            None,
            None,
            batch_size,
            eta_bn_prop=0.0,
            )

        print("Computing Dev NPMI")
        npmi = compute_npmi_at_n_during_training(
            model.get_weights(), ref_counts=dev_X.tocsc(), n=10, smoothing=0.,
        )

        print("Computing Topic diversity")
        topic_diversity = get_topic_diversity(model.get_weights(),25)


        print(f"epoch:{epoch}")
        print(f"perplexity:{perplexity:0.4f}")
        print(f"npmi:{npmi:0.4f}")
        print("topic diversity", topic_diversity)

        idx.append(epoch)
        perplexity_list.append(perplexity)
        npmi_list.append(npmi)
        TD_list.append(topic_diversity)

        # メモリ解放
        del perplexity

    df = pd.DataFrame([perplexity_list, npmi_list, TD_list], columns=idx)
    df.set_index("metric", inplace=True)
    print(df)

    df.to_csv("/home/watanabe/kd-topic-models/outputs/eval/eval.csv")




