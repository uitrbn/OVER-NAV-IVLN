import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from models.vlnbert_init import get_tokenizer

from r2r.agent_cmt_ivln_open import Seq2SeqCMTAgent

from r2r.data_utils import ImageFeaturesDB, construct_instrs
from r2r.env_ivln_openvoca import R2RBatch
from r2r.parser import parse_args
from r2r.eval_utils import compute_tour_ndtw, cal_dtw_by_dtw_py_per_ep, cal_dtw_window

from structured_memory import StructuredMemory

import sys
import pdb
import copy

# import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

def build_dataset(args, rank=0, is_test=False):
    tok = get_tokenizer(args)

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], tokenizer=tok, 
        max_instr_len=args.max_instr_len,
    )
    train_env = R2RBatch(
        feat_db, train_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        name='train', iterative=args.iterative, pre_detect='train'
    )
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], tokenizer=tok, 
            max_instr_len=args.max_instr_len
        )
        aug_env = R2RBatch(
            feat_db, aug_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            name='aug', iterative=args.iterative, pre_detect='train'
        )
    else:
        aug_env = None

    val_env_names = ['val_seen']
    if args.test or args.dataset != 'r4r':
        val_env_names.append('val_unseen')
    else:   # val_unseen of r4r is too large to evaluate in training
        assert False
        val_env_names.append('val_unseen_sampled')

    if args.submit:
        if args.dataset == 'r2r':
            val_env_names.append('test')
        elif args.dataset == 'rxr':
            assert False
            val_env_names.extend(['test_challenge_public', 'test_standard_public'])
    
    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], tokenizer=tok, 
            max_instr_len=args.max_instr_len,
        )
        val_env = R2RBatch(
            feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            name=split, iterative=args.iterative, pre_detect=split,
        )
        val_envs[split] = val_env

    return train_env, val_envs, aug_env


def train(args, train_env, val_envs, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    listner = Seq2SeqCMTAgent(args, train_env, rank=rank)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration {}".format(args.resume_file, start_iter),
                record_file
            )
       
    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            def dist_func(x, y):
                return env.hash_distance_map[x[0] * 100000 + y[0]]

            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            old_iterative = args.iterative
            args.iterative = True
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            args.iterative = old_iterative
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, all_metrics = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                
                # process tour
                # val seen len(tour) distribution: Counter({7: 45, 4: 27, 8: 24, 6: 18, 5: 15, 10: 9, 2: 6, 3: 6, 11: 6, 9: 3})
                # val unseen len(tour) distribution: Counter({100: 12, 6: 3, 15: 3, 47: 3, 60: 3, 64: 3, 93: 3, 98: 3})
                all_tours = env.tour_data
                all_tour_length = sum([len(tour) for tour in all_tours])
                all_metric_dict = {}
                for count, id in enumerate(all_metrics['instr_id']):
                    all_metric_dict[id] = {
                        metric_name: metric_values[count]
                        for metric_name, metric_values in all_metrics.items()
                    }
                all_metrics = all_metric_dict

                # compute tour metrics
                all_ndtw_by_dtw = dict()
                sr, spl, ndtw = 0, 0, 0
                t_sdtw = 0
                for tour in all_tours:
                    weight = len(tour) / all_tour_length
                    inner_sr_list, inner_spl_list = [], []
                    inner_weight_list = []
                    edited_pred_path = []
                    edited_gt_path = []
                    for i, instr_id in enumerate(tour):
                        metrics = all_metrics[instr_id]
                        scan = metrics['scan']
                        edited_pred_path += [{"position": env.hash_node_map[scan + "_" + node[0]], "episode_id": i, 'scan': scan, 'node': node[0]}
                                             for node in metrics['agent_path']]
                        edited_gt_path += [{"position": env.hash_node_map[scan + "_" + node], "episode_id": i, 'scan': scan, 'node': node}
                                           for node in metrics['gt_path']]
                        inner_weight_list.append(
                            max(metrics['gt_length'], metrics['trajectory_lengths']) / metrics['gt_length'])
                        inner_sr_list.append(metrics['success'])
                        inner_spl_list.append(metrics['spl'])
                    inner_weight_list = np.array(inner_weight_list) / sum(inner_weight_list)
                    sr += weight * (inner_weight_list * np.array(inner_sr_list)).sum()
                    spl += weight * (inner_weight_list * np.array(inner_spl_list)).sum()
                    tour_ndtw = compute_tour_ndtw(edited_pred_path, edited_gt_path, dist_func)
                    ndtw += weight * tour_ndtw

                    ndtw_by_hand = cal_dtw_window(edited_pred_path, edited_gt_path, success=None, env=env)
                    assert abs(ndtw_by_hand['nDTW'] - tour_ndtw) < 0.001

                    t_sdtw += weight * ndtw_by_hand['SDTW']

                loss_str += ', %s: %.2f' % ('t-sr', sr * 100)
                loss_str += ', %s: %.2f' % ('t-spl', spl * 100)
                loss_str += ', %s: %.2f' % ('t-nDTW', ndtw * 100)
                loss_str += ', %s: %.2f' % ('t-sDTW', t_sdtw * 100)
        if default_gpu:
            write_to_record_file(loss_str, record_file)

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "t-spl": 0., "t-sr": 0., "state":""}}

    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                # args.ml_weight = 0.2
                listner.train(1, feedback=args.feedback)

                # Train with Augmented data
                listner.env = aug_env
                # args.ml_weight = 0.2
                listner.train(1, feedback=args.feedback)

                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/RL_loss", RL_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, "
                "RL_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, RL_loss, policy_loss, critic_loss),
                record_file
            )

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, env in val_envs.items():
            def dist_func(x, y):
                return env.hash_distance_map[x[0] * 100000 + y[0]]
            listner.env = env

            # clear structured memory
            records_num = listner.structured_memory.get_records_num()
            print("Before Clearance, records number in structured memory: {}".format(records_num))
            copied_structure_memory = copy.deepcopy(listner.structured_memory)
            listner.structured_memory = StructuredMemory()
            records_num = listner.structured_memory.get_records_num()
            print("After Clearance, records number in structured memory: {}".format(records_num))

            # Get validation distance from goal under test evaluation conditions
            old_iterative = args.iterative
            args.iterative = True
            listner.test(use_dropout=False, feedback='argmax', iters=None, real_time_detect=True, use_tour_hist_inst=True)
            args.iterative = old_iterative
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, all_metrics = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

                # process tour
                all_tours = env.tour_data
                all_tour_length = sum([len(tour) for tour in all_tours])
                all_metric_dict = {}
                for count, id in enumerate(all_metrics['instr_id']):
                    all_metric_dict[id] = {
                        metric_name: metric_values[count]
                        for metric_name, metric_values in all_metrics.items()
                    }
                all_metrics = all_metric_dict

                # compute tour metrics
                all_ndtw_by_dtw = dict()
                sr, spl, ndtw = 0, 0, 0
                t_sdtw = 0
                for tour in all_tours:
                    weight = len(tour) / all_tour_length
                    inner_sr_list, inner_spl_list = [], []
                    inner_weight_list = []
                    edited_pred_path = []
                    edited_gt_path = []
                    for i, instr_id in enumerate(tour):
                        metrics = all_metrics[instr_id]
                        scan = metrics['scan']
                        edited_pred_path += [{"position": env.hash_node_map[scan + "_" + node[0]], "episode_id": i, 'scan': scan, 'node': node[0]}
                                             for node in metrics['agent_path']]
                        edited_gt_path += [{"position": env.hash_node_map[scan + "_" + node], "episode_id": i, 'scan': scan, 'node': node}
                                           for node in metrics['gt_path']]
                        inner_weight_list.append(
                            max(metrics['gt_length'], metrics['trajectory_lengths']) / metrics['gt_length'])
                        inner_sr_list.append(metrics['success'])
                        inner_spl_list.append(metrics['spl'])
                    inner_weight_list = np.array(inner_weight_list) / sum(inner_weight_list)
                    sr += weight * (inner_weight_list * np.array(inner_sr_list)).sum()
                    spl += weight * (inner_weight_list * np.array(inner_spl_list)).sum()
                    tour_ndtw = compute_tour_ndtw(edited_pred_path, edited_gt_path, dist_func)
                    ndtw += weight * tour_ndtw

                    ndtw_by_hand = cal_dtw_window(edited_pred_path, edited_gt_path, success=None, env=env)

                    assert abs(ndtw_by_hand['nDTW'] - tour_ndtw) < 0.001

                    t_sdtw += weight * ndtw_by_hand['SDTW']

                loss_str += ', %s: %.2f' % ('t-sr', sr * 100)
                loss_str += ', %s: %.2f' % ('t-spl', spl * 100)
                loss_str += ', %s: %.2f' % ('t-nDTW', ndtw * 100)
                loss_str += ', %s: %.2f' % ('t-sDTW', t_sdtw * 100)
                
                # select model by spl+sr
                if env_name in best_val:
                    if args.iterative:
                        if sr + spl >= best_val[env_name]['t-spl'] + best_val[env_name]['t-sr']:
                            best_val[env_name]['t-spl'] = spl
                            best_val[env_name]['t-sr'] = sr
                            best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                            listner.save(iter, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))
                    else:
                        if score_summary['spl'] + score_summary['sr'] >= best_val[env_name]['spl'] + best_val[env_name]['sr']:
                            best_val[env_name]['spl'] = score_summary['spl']
                            best_val[env_name]['sr'] = score_summary['sr']
                            best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                            listner.save(iter, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))
                
            # resume structured memory
            print("Before Resume, records number in structured memory: {}".format(listner.structured_memory.get_records_num()))
            listner.structured_memory = copied_structure_memory
            print("After Resume, records number in structured memory: {}".format(listner.structured_memory.get_records_num()))
        
        if default_gpu:
            listner.save(iter, os.path.join(args.ckpt_dir, "latest_dict"))

            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)


def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent = Seq2SeqCMTAgent(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        def dist_func(x, y):
            return env.hash_distance_map[x[0] * 100000 + y[0]]
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(use_dropout=False, feedback='argmax', iters=iters, real_time_detect=True)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results()
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, all_metrics = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                
                # process tour
                all_tours = env.tour_data
                all_tour_length = sum([len(tour) for tour in all_tours])
                
                # transform to tour metrics
                all_metric_dict = {}
                for count, id in enumerate(all_metrics['instr_id']):
                    all_metric_dict[id] = {
                        metric_name: metric_values[count]
                        for metric_name, metric_values in all_metrics.items()
                    }
                all_metrics = all_metric_dict
                
                # compute tour metrics
                sr, spl, ndtw = 0, 0, 0
                for tour in all_tours:
                    weight = len(tour) / all_tour_length
                    inner_sr_list, inner_spl_list = [], []
                    inner_weight_list = []
                    edited_pred_path = []
                    edited_gt_path = []
                    for i, instr_id in enumerate(tour):
                        metrics = all_metrics[instr_id]
                        scan = metrics['scan']
                        edited_pred_path += [{"position": env.hash_node_map[scan + "_" + node[0]], "episode_id": i}
                            for node in metrics['agent_path']]
                        edited_gt_path += [{"position": env.hash_node_map[scan + "_" + node], "episode_id": i}
                            for node in metrics['gt_path']]
                        inner_weight_list.append(max(metrics['gt_length'], metrics['trajectory_lengths']) / metrics['gt_length'])
                        inner_sr_list.append(metrics['success'])
                        inner_spl_list.append(metrics['spl'])
                    inner_weight_list = np.array(inner_weight_list) / sum(inner_weight_list)
                    sr += weight * (inner_weight_list * np.array(inner_sr_list)).sum()
                    spl += weight * (inner_weight_list * np.array(inner_spl_list)).sum()
                    tour_ndtw = compute_tour_ndtw(edited_pred_path, edited_gt_path, dist_func)
                    ndtw += weight * tour_ndtw
                loss_str += ', %s: %.2f' % ('t-sr', sr * 100)
                loss_str += ', %s: %.2f' % ('t-spl', spl * 100)
                loss_str += ', %s: %.2f' % ('t-nDTW', ndtw * 100)
                write_to_record_file(loss_str+'\n', record_file)

            if args.submit:
                json.dump(
                    preds,
                    open(os.path.join(args.pred_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )


def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env = build_dataset(args, rank=rank)

    if not args.test:
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank)
    else:
        valid(args, train_env, val_envs, rank=rank)
            

if __name__ == '__main__':
    main()
