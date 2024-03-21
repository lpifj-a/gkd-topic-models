from summarizer import Summarizer
import numpy as np
import pickle
import re
import string
from tqdm import tqdm
from utils import load_jsonlist, save_jsonlist, load_sparse, save_sparse, load_json, save_json
import file_handling as fh
import pickle

# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = "".join(punct_chars)
replace = re.compile("[%s]" % re.escape(punctuation))
alpha = re.compile("^[a-zA-Z_]+$")
alpha_or_num = re.compile("^[a-zA-Z_]+|[0-9_]+$")
alphanum = re.compile("^[a-zA-Z0-9_]+$")


def clean_text(
    text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False
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
    # break off single quotes at the ends of words
    # text = re.sub(r"\s\'", " ", text)
    # text = re.sub(r"\'\s", " ", text)
    # remove periods
    # text = re.sub(r"\.", "", text)
    # replace all other punctuation (except single quotes) with spaces
    # text = replace.sub(" ", text)
    # remove single quotes
    # text = re.sub(r"\'", "", text)
    # replace all whitespace with a single space
    text = re.sub(r"\s", " ", text)
    # strip off spaces on either end
    # text = text.strip()
    text = text.replace("...",".")
    return text


model = Summarizer()

train_items = fh.LazyJsonlistReader("./data/imdb/train.jsonlist")
n_train = len(train_items)
print("Found {:d} training documents".format(n_train))

teacher_embs = np.empty((n_train,1024))
nan_text_list = []  

for i, item in enumerate(tqdm(train_items)):
    text = clean_text(item["text"], strip_html=True, lower=False, keep_emails=False, keep_at_mentions=False)
    try:
        # emb = model.run_embeddings(text,max_length=10000,min_length=20,num_sentences=3,aggregate='mean')
        emb = model.run_embeddings(text,max_length=10000,min_length=20,num_sentences=3,aggregate='max')
    except:
        print("Error. Replacing text as text[:1000]")
        # emb = model.run_embeddings(text[:1000],max_length=10000,min_length=20,num_sentences=3,aggregate='mean')
        emb = model.run_embeddings(text[:1000],max_length=10000,min_length=20,num_sentences=3,aggregate='max')

    teacher_embs[i] = emb
    if emb is None:
        print("NaN is found")
        print("index:",i)
        print("text:",text)
        print("embeddings:",emb)
        nan_text_list.append(text)

with open("imdb/nan_text.pickle", mode="wb") as f:
    pickle.dump(nan_text_list, f)
    
# np.save("./embeddings/imdb/train_teacher_emb_average_pooling",teacher_embs)
np.save("./embeddings/imdb/train_teacher_emb_max_pooling",teacher_embs)

