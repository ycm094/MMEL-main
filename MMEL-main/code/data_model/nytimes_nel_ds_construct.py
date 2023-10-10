"""
    Building nel dataset
"""
import os
import json
import argparse
import pandas as pd
from nltk import word_tokenize
from ds_utils import lcss
from collections import defaultdict


root_path = '/home/yangchengmei/CODE/MMEL-main/code/'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_nes",
                        default=root_path+"data/ne2qid.json",
                        help="Path of file 'Qid-NamedEntityMapping.csv'")
    parser.add_argument("--path_dataset",
                        default=root_path+"data/dataset.json",
                        help="Path of file 'dataset.json'")
    parser.add_argument("--path_prepro",
                        default=root_path+"data/prepro_data/dm",
                        help="Path to prepro dm")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.path_prepro, exist_ok=True)

    dataset = json.load(open(args.path_dataset))

    # LCSS is used to find the overlapping part of the named entity and the sentence, and retain the referential part
    text_clean = lambda x: " ".join(x.split())
    dict_mentions = defaultdict(list)
    for k, data in dataset.items():
        data['sentence'] = text_clean(data['sentence'])
        cap_toks = set(word_tokenize(dataset[k]['sentence']))
        is_in_cap = lambda x: x in cap_toks
        for idx, ne in enumerate(data['entities']):
            ne = text_clean(data['mentions'][idx])
            mention_lcs = lcss(ne, data['sentence'])[0]
            men_toks = mention_lcs  # word_tokenize(mention_lcs)
            # men_toks = " ".join(list(filter(is_in_cap, men_toks)))
            if men_toks == ".":
                print(111111)
            if men_toks != "":
                dict_mentions[k].append((idx, men_toks))

    # Construct answers
    # read ne2qid
    """
    with open(args.path_nes, 'r+') as f:
        nes_text = f.read()
        if nes_text[:3] != 'Qid':
            f.seek(0, 0)
            f.write("Qid	Name\n" + nes_text)
    ne2qid = pd.read_csv(args.path_nes, sep='	')
    ne2qid["Name"] = [eval(x).decode(encoding="utf-8") for x in ne2qid["Name"]]
    ne2qid = dict(zip(ne2qid["Name"], ne2qid['Qid']))
    json.dump(ne2qid, open(os.path.join(args.path_prepro, "ne2qid.json"), 'w'), indent=2)
    """

    ne2qid = json.load(open(args.path_nes))

    miss_ent = []
    ds_one2mult, ds_one2one = {}, {}
    for k, mentions in dict_mentions.items():
        print(k)
        #  Get the corresponding IDs of all entities in the knowledge base
        entities = [m[0] for m in mentions]
        N = len(entities)

        entities_gt = [dataset[k]['entities'][e] for e in entities]
        
        e_gt_id = []  # [ne2qid[e] for e in entities_gt]
        for e in entities_gt:
            if e in ne2qid:
                e_gt_id.append(ne2qid[e])
            else:
                e_gt_id.append('missing')
                miss_ent.append(e)
        
        mens = [m[1] for m in mentions]

        # Answer format 1: a sentence corresponds to multiple entities
        ds_one2mult[k] = {
            "id": k,
            "sentence": dataset[k]["sentence"],
            "imgPath": dataset[k]['imgPath'],
            "mentions": mens,
            "entities": entities_gt,
            "answer": e_gt_id
        }

        # Answer format 2: a sentence corresponds to a single entity
        # Answer format 2: a sentence corresponds to a single entity
        for i, items in enumerate(zip(mens, entities_gt)):
            men, entity = items
            key = k if N == 1 else f"{k}-{i}"

            if i == 0:
                sent = dataset[k]["sentence"].replace(entity, men)
            else:
                sent = sent.replace(entity, men)

        for i, items in enumerate(zip(mens, entities_gt)):
            men, entity = items
            key = k if N == 1 else f"{k}-{i}"

            ds_one2one[key] = {
                "id": k,
                "sentence": sent,
                "imgPath": dataset[k]['imgPath'],
                "mentions": men,
                "entities": entity,
                "answer": e_gt_id[i]
            }

    miss_ent = list(set(miss_ent))
    json.dump(ds_one2mult, open(os.path.join(args.path_prepro, "nel_12n.json"), 'w'), indent=2)
    json.dump(ds_one2one, open(os.path.join(args.path_prepro, "nel_121.json"), 'w'), indent=2)
    json.dump(miss_ent, open(os.path.join(args.path_prepro, "miss_ent.json"), 'w'), indent=2)
    print("Building complete!")


if __name__ == "__main__":
    import re
    pattern = re.compile(r"/(\d)+\.")
    main()
