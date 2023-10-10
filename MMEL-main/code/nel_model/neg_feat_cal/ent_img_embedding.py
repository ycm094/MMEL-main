from transformers import BertTokenizer,BertModel
import json
import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
import h5py
logger = logging.getLogger(__name__)
from torchvision import models, transforms
from PIL import Image 
from torch.autograd import Variable
import torch.nn as nn
import clip

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
        """
        self.guid = guid
        self.words = words


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask


def read_examples_from_file(args):
    file_path = args.path_qid2abs
    examples = []
    data = json.load(open(file_path, encoding="utf-8"))
    # add
    data = {v:k for k, v in data.items()}
    # add
    keys_ordered = list(data.keys())
    json.dump(keys_ordered, open(os.path.join(args.dir_output, "ent_img_list.json"), 'w'), indent=2)
    for key in keys_ordered:
        examples.append(InputExample(guid=key, words=data[key]))
    return examples


def convert_examples_to_features(
    args,
    examples,
    max_seq_length,
    resnet,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    if resnet:
        transform = transforms.Compose([
            transforms.Scale(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize(
		        mean=[0.485, 0.456, 0.406],
		        std=[0.229, 0.224, 0.225])
	        ])

        resnet = models.resnet101(pretrained = True)
        modules = list(resnet.children())[:-2]   # delete the last fc layer.
        convnet = nn.Sequential(*modules).to(args.device)
        features = torch.FloatTensor(len(examples), 2048, 7, 7)
    else:
        model, preprocess = clip.load("ViT-B/16", device=args.device)
        features = torch.FloatTensor(len(examples), 512)
        
    count = 0
    with torch.no_grad():
        for (ex_index, example) in enumerate(examples):
            # if ex_index < 17250:
            #     continue
            print(ex_index)
            # if ex_index == 10:
            #     break
            img_path = args.dataset_path + example.guid + '.jpg' #example.words['imgPath']
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                if resnet:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = transform(img)
                    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).to(args.device)
                    y = convnet(x)
                else:
                    x = preprocess(img).unsqueeze(0).to(args.device)
                    y = model.encode_image(x)
            else:
                count += 1
                print('count : ' + str(count))
                if resnet:
                    y = torch.zeros((2048, 7, 7))
                else:
                    y = torch.zeros((512))
            
            """
            try:
                img = Image.open(img_path)
                img = transform(img)
                x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).to(args.device)
                y = convnet(x)
            except:
                y = torch.zeros((1, 2048, 1, 1))
            """
            # y = y.data.numpy()
        
            features[ex_index] = y.squeeze().cpu()
    print('count : ' + str(count))
    return features


def load_and_cache_examples(args, mode):

    # logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = read_examples_from_file(args)
    features = convert_examples_to_features( args,
            examples,
            args.max_seq_length,
            args.resnet,
        )

    # Convert to Tensors and build dataset
    dataset = TensorDataset(features)
    return dataset

def main():
    root_path = '/home/yangchengmei/CODE/MMEL-main/code/data/prepro_data/'
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--resnet",
        default=True,
        type=bool,
        # required=True,
        help="The input data.",
    )
    parser.add_argument(
        "--dataset_path",
        default='/data0/ycm/CODE/MMEL/datasets/WikiMEL/entity_imgs/',  #'dm/qid2abs.json',
        type=str,
        # required=True,
        help="The input data.",
    )
    parser.add_argument(
        "--path_qid2abs",
        default=root_path+'dm/ne2qid.json', #'dm/nel_121.json',  #'dm/qid2abs.json',
        type=str,
        # required=True,
        help="The input data.",
    )
    parser.add_argument(
        "--path_img",
        default='/home/yangchengmei/CODE/MEL-GHMFC-main/datasets/' + 'WikiMEL/entity_imgs/',  #'dm/qid2abs.json',
        type=str,
        # required=True,
        help="The input data.",
    )
    parser.add_argument( 
        "--dir_output",
        default=root_path+'nel/',
        type=str,
        # required=True,
        help="The output dir.",
    )

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--keep_accents", action="store_const", const=True, help="Set this flag if model is trained with accents."
    )
    parser.add_argument(
        "--strip_accents", action="store_const", const=True, help="Set this flag if model is trained without accents."
    )
    
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", default=False, action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device

    logger.info("Training/evaluation parameters %s", args)

    img_embeddings = load_and_cache_examples(args, mode='eval')
    h5_file = h5py.File(os.path.join(args.dir_output, "resnet_ent_img_feats.h5"), 'w')
    h5_file.create_dataset("features", data=img_embeddings.tensors[0].detach().numpy())

    logger.info("Done")


if __name__ == "__main__":
    main()
