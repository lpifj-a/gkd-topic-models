from datasets import load_dataset
import json
import file_handling as fh

train_file = 'train.jsonlist'
doc_list = []

# Download gigaword 
print("Downloading gigaword")
train = load_dataset('gigaword', split = 'train')
for l_i, line in enumerate(train["document"]):
    # Display occassional progress
    if (l_i +1) % 1000000 == 0:
        print("Processed {:d} / 3803957".format(l_i+1))
    doc = {'text': line}
    doc_list.append(doc)
print("Found %d train documents" %len(doc_list))

dev = load_dataset('gigaword', split = 'validation')
for l_i, line in enumerate(dev["document"]):
    # Display occassional progress
    if (l_i +1) % 1000000 == 0:
        print("Processed {:d} / 189,651".format(l_i+1))
    doc = {'text': line}
    doc_list.append(doc)
print("Found %d dev documents" %len(doc_list))

test = load_dataset('gigaword', split = 'test')
for l_i, line in enumerate(test["document"]):
    # Display occassional progress
    if (l_i +1) % 1000000 == 0:
        print("Processed {:d} / 1951".format(l_i+1))
    doc = {'text': line}
    doc_list.append(doc)
print("Found %d test documents" %len(doc_list))

print(doc_list[:5])
print("Found %d documents" %len(doc_list))
print("Saving processed data")
fh.write_jsonlist(doc_list, train_file)
