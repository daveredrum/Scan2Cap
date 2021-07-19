import os
import json
import torch
from tqdm import tqdm
import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor
from collections import OrderedDict

def remove_sos_eos(cap_dictionary):
    clean_cap_dictionary = {}
    for k, caps in cap_dictionary.items():
        for cap in caps:
            if isinstance(cap, type([])):
                clean_cap = cap[0].strip('sos ')
            if isinstance(cap, type('')):
                clean_cap = cap.strip('sos ')

            clean_cap = clean_cap.strip('sos')
            clean_cap = clean_cap.strip(' eos')
            clean_cap = clean_cap.strip('eos')

            if k not in list(clean_cap_dictionary.keys()):
                clean_cap_dictionary[k] = [clean_cap]
            if k in list(clean_cap_dictionary.keys()):
                clean_cap_dictionary[k].append(clean_cap)

    return OrderedDict(clean_cap_dictionary)

def prepare_corpus(scanrefer, max_len):
    corpus = {}
    for data in scanrefer:
        scene_id = data['scene_id']
        object_id = data['object_id']
        ann_id = data['ann_id']
        sample_id = '{}-{}_{}'.format(scene_id, object_id, ann_id)
        # sample_id = "{}|{}".format(data['scene_id'], data['object_id'])
        token = data['token'][:max_len]
        description = " ".join(token)

        # add start and end token
        description = "sos " + description
        description += " eos"

        if sample_id not in corpus:
            corpus[sample_id] = []

        corpus[sample_id].append(description)

    return corpus


def decode_caption(raw_caption, idx2word):
    if isinstance(raw_caption, type(torch.tensor([]))):
        decoded = ["sos"]
    else:
        decoded = []

    for token_idx in raw_caption:
        if isinstance(raw_caption, type(torch.tensor([]))):
            token_idx = token_idx.item()
        else:
            token_idx = int(token_idx)

        token = idx2word[str(token_idx)]
        decoded.append(token)
        if token == "eos":
            break

    if "eos" not in decoded: decoded.append("eos")
    decoded = " ".join(decoded)

    return decoded


def check_candidates(corpus, candidates):
    print("Number of corpus keys: ", len(corpus))
    print("Number of candidates: ", len(candidates))
    placeholder = "sos eos"
    corpus_keys = list(corpus.keys())
    candidate_keys = list(candidates.keys())
    missing_keys = [key for key in corpus_keys if key not in candidate_keys]

    print("Number of missing keys: ", len(missing_keys))
    if len(missing_keys) != 0:
        for key in missing_keys:
            candidates[key] = [placeholder]

    return candidates


def organize_candidates(corpus, candidates):
    new_candidates = {}
    for key in corpus.keys():
        new_candidates[key] = candidates[key]

    return new_candidates


def update_candidates(captions, candidates, data_dict, dataset):
    if isinstance(captions, list):
        batch_size = len(captions)
        
    else:
        batch_size, _ = captions.shape
    # dump generated captions
    for batch_id in range(batch_size):
        sample_id = data_dict['sample_id'][batch_id]
        # scene_id = data_dict['sample_id'][batch_id].split('-')[0]
        # object_id = data_dict['sample_id'][batch_id].split('-')[1].split('_')[0]
        # sample_id = '{}|{}'.format(scene_id, object_id)
        # sample_id = "{}|{}".format(data_dict['scene_id'], data['object_id'])

        caption_decoded = decode_caption(captions[batch_id], dataset.vocabulary["idx2word"])

        if sample_id not in candidates:
            candidates[sample_id] = [caption_decoded]
        else:
            candidates[sample_id].append([caption_decoded])

    return candidates


def feed_2d_cap(model, dataset, dataloader, is_eval=True):
    candidates = {}
    for data_dict in tqdm(dataloader):
        for key in data_dict:
            if not type(data_dict[key]) == type([]):
                data_dict[key] = data_dict[key].to('cuda')

        data_dict = model(data_dict, is_eval)
        captions = data_dict["lang_cap"].argmax(-1)  # batch_size, max_len - 1
        candidates = update_candidates(captions, candidates, data_dict, dataset)

    return candidates


def feed_2d_retrieval_cap(model, dataset, dataloader):
    candidates = {}
    for data_dict in tqdm(dataloader):
        captions = model(data_dict)
        candidates = update_candidates(captions, candidates, data_dict, dataset)

    return candidates

################# DEBUGGING ################
def fix_keys(corpus):
    corpus_new = {}
    for k, v in corpus.items():
        k1 = k.split('|')[0]
        k2 = k.split('|')[1]
        k_new = '{}|{}'.format(k1, k2)
        corpus_new[k_new] = v

    return corpus_new

def eval_cap(_global_iter_id,
             model,
             dataset,
             dataloader,
             phase,
             folder,
             max_len,
             mode,
             extras=False,
             is_eval=True
             ):
    # corpus
    run_config = dataset.run_config
    corpus_path = os.path.join(run_config.PATH.OUTPUT_ROOT, folder, "corpus_{}.json".format(phase))
    corpus_path_dir = os.path.dirname(corpus_path)
    os.makedirs(corpus_path_dir, exist_ok=True)

    if not os.path.exists(corpus_path):
        print("preparing corpus...")
        corpus = prepare_corpus(dataset.verified_list, max_len)
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=4)
    else:
        print("loading corpus...")
        with open(corpus_path) as f:
            corpus = json.load(f)
            corpus = corpus

    with torch.no_grad():
        if mode == 'nret':
            candidates = feed_2d_cap(model, dataset, dataloader, is_eval)
        elif mode == 'ret':
            candidates = feed_2d_retrieval_cap(model, dataset, dataloader)

    candidates = check_candidates(corpus, candidates)
    candidates = organize_candidates(corpus, candidates)
    
    pred_path = os.path.join(run_config.PATH.OUTPUT_ROOT, folder, "pred_{}.json".format(phase))
    with open(pred_path, "w") as f:
        json.dump(candidates, f, indent=4)

    if extras:
        extras = dataset.get_candidate_extras(candidates)
        with open(os.path.join(run_config.PATH.OUTPUT_ROOT, folder, "extras_{}.json".format(phase)), "w") as f:
            json.dump(extras, f, indent=4)

    # compute scores
    print("computing scores...")
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)

    # save scores
    print("saving scores...")
    score_path = os.path.join(run_config.PATH.OUTPUT_ROOT, folder, "score_{}.json".format(phase))
    with open(score_path, "w") as f:
        scores = {
            "bleu-1": [float(s) for s in bleu[1][0]],
            "bleu-2": [float(s) for s in bleu[1][1]],
            "bleu-3": [float(s) for s in bleu[1][2]],
            "bleu-4": [float(s) for s in bleu[1][3]],
            "cider": [float(s) for s in cider[1]],
            "rouge": [float(s) for s in rouge[1]],
            "meteor": [float(s) for s in meteor[1]],
        }
        json.dump(scores, f, indent=4)

    return bleu, cider, rouge, meteor
