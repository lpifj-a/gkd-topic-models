from io import StringIO
from scipy.io import mmread
from sklearn.model_selection import train_test_split
import file_handling as fh
from scipy import sparse

with open("bbc.terms") as f:
    vocab = [line.rstrip() for line in f.readlines()]
    print("vocab size:",len(vocab))
    fh.write_to_json(vocab, "train.vocab.json")

m = mmread("bbc.mtx")
m = m.A.T
print("document-term matrix:",m.shape)
n_total = m.shape[0]
print("Found {} documents".format(n_total))

doc_ids = list(range(n_total))
train_ids, rem_ids = train_test_split(doc_ids, train_size=0.7, random_state=42)
test_ids, dev_ids = train_test_split(rem_ids, test_size=0.5, random_state=42)

print("train documents:",len(train_ids))
print("dev documents:",len(dev_ids))
print("test documents:",len(test_ids))

train_X = m[train_ids]
test_X = m[test_ids]
dev_X = m[dev_ids]

sparse_train_X = sparse.csr_matrix(train_X)
sparse_test_X = sparse.csr_matrix(test_X)
sparse_dev_X = sparse.csr_matrix(dev_X)

fh.save_sparse(sparse_train_X, "train.npz")
fh.save_sparse(sparse_test_X, "test.npz")
fh.save_sparse(sparse_dev_X, "dev.npz")

fh.write_to_json(train_ids, "train.ids.json")
fh.write_to_json(test_ids, "test.ids.json")
fh.write_to_json(dev_ids, "dev.ids.json")