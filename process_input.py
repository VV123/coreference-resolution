from bert import tokenization
import json

genre = 'nw'
model_name = 'spanbert_large'
data = {
    'doc_key': genre,
    'sentences': [["[CLS]"]],
    'speakers': [["[SPL]"]],
    'clusters': [],
    'sentence_map': [0],
    'subtoken_map': [0],
}

# Determine Max Segment
max_segment = None
for line in open('experiments.conf'):
    if line.startswith(model_name):
        max_segment = True
    elif line.strip().startswith("max_segment_len"):
        if max_segment:
            max_segment = int(line.strip().split()[-1])
            break

tokenizer = tokenization.FullTokenizer(vocab_file="cased_config_vocab/vocab.txt", do_lower_case=False)
subtoken_num = 0
text = []
with open('input.txt', 'r') as f:
    for l in f:
        l = l.strip('\n')
        text.append(l)

out = open("sample.in.json", 'w') 
for sent_num, line in enumerate(text):
    raw_tokens = line.split()
    tokens = tokenizer.tokenize(line)
    if len(tokens) + len(data['sentences'][-1]) >= max_segment:
        data['sentences'][-1].append("[SEP]")
        data['sentences'].append(["[CLS]"])
        data['speakers'][-1].append("[SPL]")
        data['speakers'].append(["[SPL]"])
        data['sentence_map'].append(sent_num - 1)
        data['subtoken_map'].append(subtoken_num - 1)
        data['sentence_map'].append(sent_num)
        data['subtoken_map'].append(subtoken_num)

    ctoken = raw_tokens[0]
    cpos = 0
    for token in tokens:
        data['sentences'][-1].append(token)
        data['speakers'][-1].append("-")
        data['sentence_map'].append(sent_num)
        data['subtoken_map'].append(subtoken_num)
        
        if token.startswith("##"):
            token = token[2:]
        if len(ctoken) == len(token):
            subtoken_num += 1
            cpos += 1
            if cpos < len(raw_tokens):
                ctoken = raw_tokens[cpos]
        else:
            ctoken = ctoken[len(token):]

    data['sentences'][-1].append("[SEP]")
    data['speakers'][-1].append("[SPL]")
    data['sentence_map'].append(sent_num - 1)
    data['subtoken_map'].append(subtoken_num - 1)


    out.write(json.dump(data, out, sort_keys=True)+'\n')
out.close()
