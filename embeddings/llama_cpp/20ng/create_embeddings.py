from langchain_community.embeddings import LlamaCppEmbeddings
import numpy as np
import pickle
import re
import string
from tqdm import tqdm
from utils import load_jsonlist, save_jsonlist, load_sparse, save_sparse, load_json, save_json

# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = "".join(punct_chars)
replace = re.compile("[%s]" % re.escape(punctuation))
alpha = re.compile("^[a-zA-Z_]+$")
alpha_or_num = re.compile("^[a-zA-Z_]+|[0-9_]+$")
alphanum = re.compile("^[a-zA-Z0-9_]+$")


def clean_text(
    text, strip_html=False, lower=False, keep_emails=False, keep_at_mentions=False
):
    # remove html tags
    if strip_html:
        text = re.sub(r"<[^>]+>", "", text)
    else:
        # replace angle brackets
        text = re.sub(r"<", "(", text)
        text = re.sub(r">", ")", text)
    # lower case
    if lower:
        text = text.lower()
    # eliminate email addresses
    if not keep_emails:
        text = re.sub(r"\S+@\S+", " ", text)
    # eliminate @mentions
    if not keep_at_mentions:
        text = re.sub(r"\s@\S+", " ", text)
    # replace underscores with spaces
    text = re.sub(r"_", " ", text)
    text = re.sub(r"\s", " ", text)
    text = text.replace("\n","")
    text = text.replace("...",".")
    return text

data = load_jsonlist("./data/20ng/aligned/dev/train.jsonlist")
print("Number of data:",len(data))

# llama = LlamaCppEmbeddings(model_path="./embeddings/llama.cpp/20ng/models/llama-2-13b-chat/ggml-model-q4_0.gguf",n_gpu_layers=42,n_batch=1000,n_ctx=20000,n_threads=4)
llama = LlamaCppEmbeddings(model_path="./embeddings/llama-cpp/20ng/models/Yarn-Llama-2-13B-128K-GGUF.gguf",n_gpu_layers=35,n_batch=500,n_ctx=25000,n_threads=None)

teacher_embs = np.empty((len(data),5120))
long_text_idx = []

for i,d in enumerate(tqdm(data)):
    text = d["text"]
    text = clean_text(text, strip_html=True, lower=False, keep_emails=False, keep_at_mentions=False)

    try:
        emb = llama.embed_query(text)
        teacher_embs[i] = np.array(emb)
    except:
        print("error!")
        print("index:",i)

np.save("./embeddings/llama_cpp/20ng/train_teacher_emb",teacher_embs)

