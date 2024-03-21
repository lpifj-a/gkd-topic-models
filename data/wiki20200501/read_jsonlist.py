import json
import file_handling as fh
from tqdm import tqdm

file = "/home/watanabe/kd-topic-models/data/wiki20200501/train.jsonlist"

print("Loading datasets")
lines = fh.read_text(file)
doc_len_list = []
n_doc = len(lines)
print("document size:", n_doc)
for doc in tqdm(lines):
    doc_len = len(doc)
    doc_len_list.append(doc_len)

mean = sum(doc_len_list)/len(doc_len_list)
print("Average doc length",mean)