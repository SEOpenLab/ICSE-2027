import os
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import json
import torch
import pickle
import collections

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,TFAutoModelForCausalLM, AutoConfig, AutoModel
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import BartTokenizer, AutoModelForSeq2SeqLM, BartConfig
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig

class InputExample(object):
    def __init__(self, text, unique_id):
        self.text = text
        self.unique_id = unique_id

class InputFeatures(object):
    def __init__(self, tokens, unique_id, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.unique_id = unique_id

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def read_examples(text_file):
    examples = []
    unique_id = 0

    with open(text_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line: 
                break

            text = line.strip().split('\t')[-1]
            examples.append(InputExample(text=text, unique_id=unique_id))
            unique_id += 1
    return examples

def convert_examples_to_features(examples, seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        cand_tokens = tokenizer.tokenize(example.text)
        #print(cand_tokens)
        if len(cand_tokens) > seq_length - 2: 
            ## Account for [CLS] and [SEP] with "- 2"
            cand_tokens = cand_tokens[0:(seq_length - 2)] 

        tokens = []
        input_type_ids = []

        tokens.append(tokenizer.cls_token) #[CLS] #-id-101
        input_type_ids.append(0)
        for token in cand_tokens:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append(tokenizer.sep_token)  #[SEP] #-id-102
        input_type_ids.append(0)

        input_ids  = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(tokenizer.pad_token_id) #[PAD] #-0
            input_mask.append(0)
            input_type_ids.append(0)

        #assert len(input_ids) == seq_length
        #assert len(input_mask) == seq_length
        #assert len(input_type_ids) == seq_length
        features.append(InputFeatures(tokens=tokens, unique_id=example.unique_id, input_ids=input_ids, input_mask=input_mask, input_type_ids=input_type_ids))
        print("#" * 20)
        print(tokens)
        print("#" * 20)
        print("#" * 20)
        print(input_ids)
        print("#" * 20)
        print(input_mask)
        print("#" * 20)
    return features

def get_max_seq_length(samples, tokenizer):
    max_seq_len = -1
    for sample in samples:
        cand_tokens = tokenizer.tokenize((sample.text))
        cur_len = len(cand_tokens)
        if cur_len > max_seq_len:
            max_seq_len = cur_len

    # *************************************
    if max_seq_len > model_max_seq_length:
        max_seq_len = model_max_seq_length
    # *************************************

    return max_seq_len

def save_features(model, tokenizer, device):
    # convert data to ids
    examples = read_examples(text_dataset)
    features = convert_examples_to_features(examples=examples, seq_length=(get_max_seq_length(examples, tokenizer)), tokenizer=tokenizer)

    # extract and write features
    all_input_ids       = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask      = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_indices = torch.arange(all_input_ids.size(0), dtype=torch.long) # gives => tensor([0,1, 2, ... (num_samples - 1) ])
    eval_dataset        = TensorDataset(all_input_ids, all_input_mask, all_example_indices)
    eval_dataloader     = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=batchsize)

    pbar = tqdm(total=len(examples)//batchsize)
    with open(json_features, "w") as writer:
        with torch.no_grad():
            for input_ids, input_mask, example_indices in eval_dataloader: # batch_sized input_ids, input_mask, example_indices tensor
                input_ids   = input_ids.to(device)    # batch_sized input_ids tensor
                input_mask  = input_mask.to(device)   # batch_sized input_mask tensor

                attention_mask_mean = input_mask.clone()

                attention_mask_mean[:, 0] = 0

                sep_index = attention_mask_mean.sum(dim=1) - 1
                for i in range(attention_mask_mean.size(0)):
                    attention_mask_mean[i, sep_index[i]] = 0
                # =======================================================================

                # =======================================================================


                all_outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask) 
                enc_layers  = all_outputs.hidden_states 
                #print("***************************************************")
                #print(model_checkpoint, " => Num layers:", len(enc_layers))
                #print("***************************************************")


                for iter_index, example_index in enumerate(example_indices):
                    # for every feature in batch => tokens, input_ids, input_mask => features[example_index.item()]
                    feature     = features[example_index.item()] # example_indices are i,j,k, ... till batch_size
                    unique_id   = int(feature.unique_id)

                    all_output_features = []
                    all_layers = []
                    for layer_index in range(len(enc_layers)):
                        layer_output = enc_layers[int(layer_index)]  # layer   layer_index (#0, #1, #2 ... max_layers)
                        layer_feat_output = layer_output[iter_index] # feature iter_index 

                        mask = attention_mask_mean[iter_index]            # [seq_len]
                        input_mask_expanded = mask.unsqueeze(-1).expand(layer_feat_output.size()).float()
                        mean_embedding = torch.sum(layer_feat_output * input_mask_expanded, 0) / torch.clamp(input_mask_expanded.sum(0), min=1e-9)


                        # ==================================================================
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index
                        layers["values"] = [round(v.item(), 6) for v in mean_embedding]
                        all_layers.append(layers)

                    out_features = collections.OrderedDict()
                    out_features["token"] = "MEAN"
                    out_features["layers"] = all_layers
                    all_output_features.append(out_features)

                    output_json = collections.OrderedDict()
                    output_json["linex_index"] = unique_id
                    output_json["features"] = all_output_features
                    writer.write(json.dumps(output_json) + "\n")

                pbar.update(1)
    pbar.close()
    print('written features to %s'%(json_features))


if __name__ == '__main__':

    task_codes    = ['IDN', 'CPX', 'AST'] # TODO: put tasks in tasks.config ['KTX', 'IDN', 'LEN', 'TYP', 'REA', 'JBL', 'SRI', 'SRK', 'SCK', 'OCU', 'VCU', 'CSC', 'MXN', 'CPX', 'NPT']
    shuffle_kinds = ['ORIG']
    label_counts  = ['1k'] #['100', '1k', '10k']

    model_checkpoints = { # TODO: put models in models.config OR make a models class

        "BERT-base-cased":          "google-bert/bert-base-cased", 

    }

    model_max_seq_lengths = {

        "BERT-base-cased":   512,


    }


    for task_code in task_codes:
        for shuffle_kind in shuffle_kinds:
            for model_checkpoint in list(model_checkpoints.keys()):
                for label_count in label_counts:
                    print("********")
                    print(f"Processing for task >> {task_code} >> {shuffle_kind}:{model_checkpoint} for {label_count}")
                    print("********")

                    text_dataset  = sys.path[0] + '/data/datasets_'+ task_code +'/'+ task_code +'_'+ shuffle_kind +'_'+ label_count +'.txt'
                    json_features = sys.path[0] + '/data/datasets_'+ task_code +'/'+ shuffle_kind +'/'+ model_checkpoint +'_features_'+ label_count +'.json'

                    if not os.path.exists(json_features):
                        path = Path(json_features)
                        path.parent.mkdir(parents=True, exist_ok=True) 

                    # *******************************************************

                    modelname = model_checkpoints.get(model_checkpoint, None)
                    model_max_seq_length = model_max_seq_lengths.get(model_checkpoint, None)

                    print(modelname)

                    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    batchsize = 8 #8 for 512 tokens 4 for 1024 tokens # TODO: get batch size from args
                    print("Using device: ", device)


                    #Additional Info when using cuda
                    if device.type == 'cuda':
                        cur_device = torch.cuda.current_device()
                        print(torch.cuda.get_device_name(cur_device))
                        print('Memory Usage:')
                        print('Allocated:', round(torch.cuda.memory_allocated(cur_device)/1024**3,1), 'GB')
                        print('Cached:   ', round(torch.cuda.memory_reserved(cur_device)/1024**3,1), 'GB')


                    if model_checkpoint in ["BERT-base-uncased"]:
                        model_path = sys.path[0] + "/tmp/" + modelname
                        config    = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
                        tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
                        model     = AutoModel.from_pretrained(model_path, config=config)
                        model.resize_token_embeddings(len(tokenizer)) #tokenizer 和 model 是配套的, this is  meaningless

                    elif model_checkpoint in ["BERT-base-cased", "JavaBERT-KIEL"]:
                        model_path = sys.path[0] + "/tmp/" + modelname                    
                        config    = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        model     = AutoModel.from_pretrained(model_path, config=config)
                        model.resize_token_embeddings(len(tokenizer)) #tokenizer 和 model 是配套的, this is  meaningless

                    print("#" * 20)
                    print(modelname)
                    print("#" * 20)
                    print("Model Size:", round(float(model.num_parameters() / 1000000), 2), "Million" )
                    print("=" * 20)
                    print("Num. Hidden Layers:\t", model.config.num_hidden_layers)
                    print("Num. Attention Heads:\t", model.config.num_attention_heads)
                    print("Embedding Hiddden Size:\t", model.config.hidden_size)
                    print("tokenizer.cls_token:\t", tokenizer.cls_token)
                    print("tokenizer.cls_token_id:\t", tokenizer.cls_token_id)
                    print("tokenizer.sep_token:\t", tokenizer.sep_token)
                    print("tokenizer.sep_token_id:\t", tokenizer.sep_token_id)      
                    print("tokenizer.pad_token:\t", tokenizer.pad_token)
                    print("tokenizer.pad_token_id:\t", tokenizer.pad_token_id)                                   
                    print("=" * 20)
                    print("Vocabulary Size:\t", model.config.vocab_size)
                    print("Tokenizer Length:\t", len(tokenizer))
                    print("=" * 20)

                    model.to(device)
                    model.eval()
                    save_features(model, tokenizer, device)
                    print("********")
                    print("\n" * 2)



