"""
    search kg for candidates by name
"""
import argparse
import json
import os
from os.path import join, exists
from fuzzywuzzy import process
from multiprocessing.pool import Pool
import logging

logger = logging.getLogger(__name__)

root_path = '/home/yangchengmei/CODE/MEL-GHMFC-main/code/'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_dm",
                        default=root_path+"data/prepro_data/dm",
                        help="Dir to dm")
    parser.add_argument("--dir_prepro",
                        default=root_path+"data/prepro_data/nel",
                        help="Dir to NEL prepro")
    parser.add_argument("--num_search",
                        default=100,
                        type=int,
                        help="Number of search results")
    return parser.parse_args()


def run(m, ne_list, num_search):
    return process.extract(m, ne_list, limit=num_search)


def main():
    args = parse_args()
    logger.info(f"Search candidates with fuzzywuzzy (num: {args.num_search}).")
    ne2qid = json.load(open(os.path.join(args.dir_dm, "ne2qid.json")))
    nel_one2one = json.load(open(os.path.join(args.dir_dm, "nel_121.json")))

    path_search_tmp = join(args.dir_prepro, f"search_tmp{args.num_search}.json")

    if not exists(path_search_tmp):
        ne_list = list(ne2qid.keys())# [:200]
        key_list = list(nel_one2one.keys())
        mentions = [nel_one2one[key]["mentions"] for key in key_list]#[:1000]

        pool = Pool(10)
        search_res = [pool.apply_async(run, (m, ne_list, args.num_search,)) for m in mentions]
        search_tmp = []  # [sr.get() for sr in search_res]
        for i in range(len(search_res)):
            print(i)
            sr = search_res[i]
            search_tmp.append(sr.get())
        json.dump({
            'search_res': search_tmp,
            'key_ordered': key_list
        }, open(path_search_tmp, 'w'), indent=2)
    else:
        tmp = json.load(open(path_search_tmp))
        key_list = tmp['key_ordered']
        search_tmp = tmp["search_res"]


    N = len(key_list)
    men2qids = dict()
    for i in range(N):
        print(i)
        mentions = nel_one2one[key_list[i]]["mentions"]  # ["entities"]
        qids = []
        for item in search_tmp[i]:
            ne = item[0]
            qids.append(ne2qid[ne])

        men2qids[mentions] = qids

    json.dump(men2qids, open(os.path.join(args.dir_prepro, f"search_top{args.num_search}.json"), 'w'), indent=2)


if __name__ == '__main__':
    main()