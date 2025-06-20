import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim

from transformers import BertTokenizer, BertForTokenClassification
import argparse
import gc
import numpy as np
import os
import sys
from tqdm import tqdm
import time
from datetime import datetime
import random
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from dataset import HateXplainDataset
from utils import get_device, GetLossAverage, save_checkpoint, add_tokens_to_tokenizer
from prefinetune_utils import prepare_gts, make_masked_rationale_label, add_pads
from module import BertForTCwMRP


def get_args_1():
    parser = argparse.ArgumentParser(description='')

    # DATASET
    parser.add_argument('--dir_hatexplain', type=str, default="./dataset", help='the root directiory of the dataset')
    
    # PRETRAINED MODEL
    model_choices = ['bert-base-uncased']
    parser.add_argument('--pretrained_model', default='bert-base-uncased', choices=model_choices, help='a pre-trained bert model to use')  

    # TRAIN
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--val_int', type=int, default=945)  
    parser.add_argument('--patience', type=int, default=3)

    ## Pre-Finetuing Task
    parser.add_argument('--intermediate', choices=['mrp', 'rp'], required=True, help='choice of an intermediate task')

    ## Masked Ratioale Prediction 
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--n_tk_label', type=int, default=2)
    
    ##### ADDED PARTS #####
    
    # Strategic Masking
    parser.add_argument('--strategic_masking', action='store_true', help='if True, use strategic masking')
    
    # Contrastive Loss
    parser.add_argument('--contrastive_loss', action='store_true', help='if True, use contrastive loss')
    parser.add_argument('--contrastive_weight', type=float, default=0.1, help='weight for contrastive loss')
    parser.add_argument('--contrastive_temperature', type=float, default=0.1, help='temperature for contrastive loss')
    
    #### END OF ADDED PARTS ####

    args = parser.parse_args()
    return args   

def get_prob_acc(args, gts_tensor, ori_gts, out_logits):
    batch_pred_probs = []
    batch_gt_probs = []
    batch_bn_gts = []
    batch_bn_preds = []
    batch_prob_acc = {'prob_acc':[], 'cls_acc':[]}
    
    for gt, ori_gt, logit in zip(gts_tensor, ori_gts, out_logits):
        out_prob = F.softmax(logit, dim=1)
        out_prob_T = torch.transpose(out_prob, 0, 1)
        gt_T = torch.transpose(gt, 0, -1)

        pred_prob = out_prob_T[1].tolist()
        gt_prob = gt_T.tolist()

        assert len(pred_prob) == len(gt_prob), "[!] get_prob_acc() | #gt != #pred"
        batch_pred_probs.append(pred_prob)
        batch_gt_probs.append(gt_prob)

        tmp = 0
        dif = 0
        bn_pred = []
        bn_gt = gt_prob
        for g, o, p in zip(gt_prob, ori_gt, pred_prob):
            dif += abs(o - p)
            p_cls = int(p >= 0.5)
            bn_pred.append(p_cls)

            if g == p_cls:
                tmp += 1

        batch_bn_gts += bn_gt
        batch_bn_preds += bn_pred
        c_acc = tmp / len(gt_prob)
        p_acc = 1.0 - dif / len(gt_prob)
        batch_prob_acc['prob_acc'].append(p_acc)
        batch_prob_acc['cls_acc'].append(c_acc)

    return batch_pred_probs, batch_gt_probs, batch_prob_acc, batch_bn_gts, batch_bn_preds
        

def evaluate(args, model, dataloader, tokenizer, emb_layer, mlb, token_ambiguity=None):
    all_pred_clses, all_pred_clses_masked, all_gts, all_gts_masked_only = [], [], [], []
    losses = []
    consumed_time = 0

    model.eval()
    if args.intermediate == 'mrp':
        emb_layer.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="EVAL | # {}".format(args.n_eval), mininterval=0.01)):
            in_tensor = tokenizer(batch[0], return_tensors='pt', padding=True)
            in_tensor = in_tensor.to(args.device)
            max_len = in_tensor['input_ids'].shape[1]

            if args.intermediate == 'rp':
                gts = prepare_gts(args, max_len, batch[2])
                gts_tensor = torch.tensor(gts).to(args.device)

                start_time = time.time()
                out_tensor = model(**in_tensor, labels=gts_tensor)
                consumed_time += time.time() - start_time   

            elif args.intermediate == 'mrp':
                gts = prepare_gts(args, max_len, batch[2])
                
                #### ADDED PARTS ####
                
                # Get tokenized input for strategic masking
                tokens = None
                if args.strategic_masking:
                    tokens = [tokenizer.tokenize(text) for text in batch[0]]
                
                masked_idxs, label_reps, masked_gts = make_masked_rationale_label(args, gts, emb_layer, input_tokens=tokens, token_ambiguity=token_ambiguity)
                    
                #### END OF ADDED PARTS ####
                
                gts_pad, masked_gts_pad, label_reps = add_pads(args, max_len, gts, masked_gts, label_reps)

                label_reps = torch.stack(label_reps).to(args.device)
                gts_tensor = torch.tensor(masked_gts_pad).to(args.device)
                in_tensor['label_reps'] = label_reps

                start_time = time.time()
                out_tensor = model(**in_tensor, labels=gts_tensor)
                consumed_time += time.time() - start_time

            loss = out_tensor.loss.item()
            logits = out_tensor.logits
            pred_probs = F.softmax(logits, dim=2)

            losses.append(loss)
            
            if args.intermediate == 'rp':
                pred_probs = pred_probs.detach().cpu().numpy()
                pred_clses = np.argmax(pred_probs, axis=2)
                pred_clses = pred_clses.tolist()
                all_pred_clses += pred_clses
                all_gts += gts
                
            elif args.intermediate == 'mrp':
                pred_probs = F.softmax(logits, dim=2)
                pred_clses_pad, pred_clses_wo_pad, pred_clses_masked, gts_masked_only = [], [], [], []
                for pred_prob, idxs, gt in zip(pred_probs, masked_idxs, gts):
                    pred_cls = [p.index(max(p)) for p in pred_prob.tolist()]
                    pred_clses_pad += pred_cls

                    if len(pred_cls) == len(gt):
                        pred_cls_wo_pad = pred_cls
                    else:
                        pred_cls_wo_pad = pred_cls[(len(pred_cls)-len(gt)):]
                    pred_clses_wo_pad += pred_cls_wo_pad

                    pred_cls_masked = [pred_cls[i] for i in idxs]
                    gt_masked_only = [gt[i] for i in idxs]
                    pred_clses_masked += pred_cls_masked
                    gts_masked_only += gt_masked_only
                    all_gts += gt    
                
                all_pred_clses += pred_clses_wo_pad
                all_pred_clses_masked += pred_clses_masked
                all_gts_masked_only += gts_masked_only

    loss_avg = sum(losses) / len(dataloader)
    time_avg = consumed_time / len(dataloader)

    if args.intermediate == 'rp':
        all_gts = mlb.fit_transform(all_gts)
        all_pred_clses = mlb.fit_transform(all_pred_clses)
    acc = [accuracy_score(all_gts, all_pred_clses)]
    f1 = [f1_score(all_gts, all_pred_clses, average='macro')]
    if args.intermediate == 'mrp':
        acc.append(accuracy_score(all_gts_masked_only, all_pred_clses_masked))
        f1.append(f1_score(all_gts_masked_only, all_pred_clses_masked, average='macro'))

    return losses, loss_avg, time_avg, acc, f1


def train(args):
    # Load the model
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    tokenizer = add_tokens_to_tokenizer(args, tokenizer)
    if args.intermediate == 'rp':
        model = BertForTokenClassification.from_pretrained(args.pretrained_model)
        emb_layer = None
    elif args.intermediate == 'mrp':
        model = BertForTCwMRP.from_pretrained(args.pretrained_model)
        emb_layer = nn.Embedding(args.n_tk_label, 768)
        model.config.output_attentions=True
    
    model.resize_token_embeddings(len(tokenizer))

    # Define dataloader
    train_dataset = HateXplainDataset(args, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = HateXplainDataset(args, 'val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    #### ADDED PARTS ####
    
    token_ambiguity = None
    if args.intermediate == 'mrp' and args.strategic_masking:
        token_ambiguity_path = os.path.join(args.dir_hatexplain, 'token_statistics.json')
        if os.path.exists(token_ambiguity_path):
            import json
            with open(token_ambiguity_path, 'r') as f:
                token_ambiguity = json.load(f)
        else:
            from prefinetune_utils import calculate_token_statistics
            print("Calculating token statistics for strategic masking...")
            token_ambiguity = calculate_token_statistics(train_dataset, tokenizer, save_path=args.dir_hatexplain)
    
    get_contrastive_loss = None
    if args.contrastive_loss:
        from prefinetune_utils import (
            generate_contrastive_pairs, 
            extract_token_representations, 
            create_contrastive_batch_tensors,
            compute_contrastive_loss
        )
        get_contrastive_loss = GetLossAverage()
        
    #### END OF ADDED PARTS ####
    
    get_tr_loss = GetLossAverage()
    mlb = MultiLabelBinarizer()

    if args.intermediate == 'mrp':
        optimizer = optim.RAdam(list(emb_layer.parameters())+list(model.parameters()), lr=args.lr, betas=(0.9, 0.99))
        emb_layer.to(args.device)
        emb_layer.train()
    else:
        optimizer = optim.RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    
    model.to(args.device)
    model.train()

    # configuration = model.config
    log = open(os.path.join(args.dir_result, 'train_res.txt'), 'a')
    
    tr_losses = []
    val_losses = []
    val_cls_accs = []
    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(train_dataloader, desc="TRAIN | Epoch: {}".format(epoch), mininterval=0.01)):  # data: (post_words, target_rat, post_id)
            in_tensor = tokenizer(batch[0], return_tensors='pt', padding=True)
            max_len = in_tensor['input_ids'].shape[1]
            
            optimizer.zero_grad()

            if args.intermediate == 'rp':  
                args.contrastive_loss = False
                in_tensor = in_tensor.to(args.device)
                gts = prepare_gts(args, max_len, batch[2])
                gts_tensor = torch.tensor(gts).long().to(args.device)
                out_tensor = model(**in_tensor, labels=gts_tensor)
                
            elif args.intermediate == 'mrp':
                in_tensor = in_tensor.to(args.device)
                gts = prepare_gts(args, max_len, batch[2])
                
                ##### ADDED PARTS ####
                
                # Get tokenized input for strategic masking and contrastive learning
                tokens = None
                positive_pairs, negative_pairs = None, None
                
                if args.strategic_masking or args.contrastive_loss:
                    tokens = [tokenizer.tokenize(text) for text in batch[0]]
                    
                # Generate contrastive pairs
                if args.contrastive_loss:
                    positive_pairs, negative_pairs = generate_contrastive_pairs(tokens, gts)
                    
                masked_idxs, label_reps, masked_gts = make_masked_rationale_label(args, gts, emb_layer, input_tokens=tokens, token_ambiguity=token_ambiguity)
                
                #### END OF ADDED PARTS ####
                
                gts_pad, masked_gts_pad, label_reps = add_pads(args, max_len, gts, masked_gts, label_reps)

                label_reps = torch.stack(label_reps).to(args.device)
                gts_tensor = torch.tensor(masked_gts_pad).to(args.device)
                in_tensor['label_reps'] = label_reps
                out_tensor = model(**in_tensor, labels=gts_tensor, output_hidden_states=True)
                
            #### ADDED PARTS ####
            
            # Compute contrastive loss if enabled
            contrastive_loss = 0
            if args.contrastive_loss and positive_pairs is not None and negative_pairs is not None:
                try:
                    # Extract representation
                    representations = extract_token_representations(
                        model_outputs=out_tensor,
                        attention_mask=in_tensor['attention_mask'],
                        rationale_labels=torch.tensor(gts_pad).to(args.device),
                        positive_pairs=positive_pairs,
                        negative_pairs=negative_pairs
                        )
                    
                    # Create batched tensors for contrastive loss
                    contrastive_tensors = create_contrastive_batch_tensors(
                        representations['positive_pair_embeddings'],
                        representations['negative_pair_embeddings'],
                        args.device
                    )
                    
                    if contrastive_tensors is not None:
                        contrastive_loss = compute_contrastive_loss(
                            contrastive_tensors['anchors'],
                            contrastive_tensors['pairs'],
                            contrastive_tensors['labels'],
                            temperature=args.contrastive_temperature if hasattr(args, 'contrastive_temperature') else 0.1
                        )
                        
                        if get_contrastive_loss:
                            get_contrastive_loss.add(contrastive_loss)
                except Exception as e:
                    print(f"Warning: Contrastive loss computation failed: {e}")
                    contrastive_loss = 0
                
            # Combine losses
            total_loss = out_tensor.loss
            if args.contrastive_loss and contrastive_loss != 0:
                # Add contrastive loss with weighting
                contrastive_weight = args.contrastive_weight if hasattr(args, 'contrastive_weight') else 0.1
                total_loss += contrastive_weight * contrastive_loss
                
            #### END OF ADDED PARTS ####
            
            total_loss.backward()
            optimizer.step()
            get_tr_loss.add(total_loss)

            # validation
            if i == 0 or (i+1) % args.val_int == 0:
                _, val_loss, val_time, acc, f1 = evaluate(args, model, val_dataloader, tokenizer, emb_layer, mlb, token_ambiguity=token_ambiguity)
                
                args.n_eval += 1
                model.train()

                val_losses.append(val_loss)
                tr_loss = get_tr_loss.aver()
                tr_losses.append(tr_loss) 
                get_tr_loss.reset()  
                
                ### ADDED PARTS ####
                
                # Log contrastive loss if enabled
                contrastive_loss_avg = 0
                if get_contrastive_loss:
                    contrastive_loss_avg = get_contrastive_loss.aver()
                    get_contrastive_loss.reset()
                    
                #### END OF ADDED PARTS ####
                
                print("[Epoch {} | Val #{}]".format(epoch, args.n_eval))
                print("* tr_loss: {}".format(tr_loss))
                if args.contrastive_loss:
                    print("* tr_contrastive_loss: {}".format(contrastive_loss_avg))
                print("* val_loss: {} | val_consumed_time: {}".format(val_loss, val_time))
                print("* acc: {} | f1: {}".format(acc[0], f1[0]))
                if args.intermediate == 'mrp':
                    print("* acc about masked: {} | f1 about masked: {}".format(acc[1], f1[1]))
                print('\n')

                log.write("[Epoch {} | Val #{}]\n".format(epoch, args.n_eval))
                log.write("* tr_loss: {}\n".format(tr_loss))
                if args.contrastive_loss:
                    log.write("* tr_contrastive_loss: {}\n".format(contrastive_loss_avg))
                log.write("* val_loss: {} | val_consumed_time: {}\n".format(val_loss, val_time))
                log.write("* acc: {} | f1: {}\n".format(acc[0], f1[0]))
                if args.intermediate == 'mrp':
                    log.write("* acc about masked: {} | f1 about masked: {}\n".format(acc[1], f1[1]))
                log.write('\n')

                save_checkpoint(args, val_losses, emb_layer, model)
                
            if args.waiting > args.patience:
                print("early stopping")
                break
        if args.waiting > args.patience:
            break 

    log.close()


if __name__ == '__main__':
    args = get_args_1()
    args.test = False
    args.device = get_device()

    lm = '-'.join(args.pretrained_model.split('-')[:-1])

    now = datetime.now()
    args.exp_date = now.strftime('%m%d-%H%M')
    args.exp_name = args.exp_date + '_'+ lm + '_' + args.intermediate +  "_"  + str(args.lr) + "_" + str(args.batch_size) + "_" + str(args.val_int)
    
    #### ADDED PARTS ####
    
    args.multitask = False
    
    if args.strategic_masking:
        args.exp_name += "_strategic_masking"
        
    if args.contrastive_loss:
        args.exp_name += "_contrastive" + "_" + str(args.contrastive_weight) + "_" + str(args.contrastive_temperature)
        
    #### END OF ADDED PARTS ####
    
    dir_result = os.path.join("finetune_1st", args.exp_name)
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)

    print("Checkpoint path: ", dir_result)
    
    args.dir_result = dir_result
    args.waiting = 0
    args.n_eval = 0

    gc.collect()
    torch.cuda.empty_cache()
    
    #### ADDED PARTS ####
    
    # Set random seed for reproducibility
    from utils import set_seed
    set_seed(40)
    
    ##### END OF ADDED PARTS ####

    train(args)