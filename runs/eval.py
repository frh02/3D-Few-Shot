"""Evaluating functions for Few-shot 3D Point Cloud Semantic Segmentation

Author: Zhao Na, 2020
"""
import os
import numpy as np
from datetime import datetime
import ast
import argparse
from dataloaders.loader_ova import MyDataset, MyTestDataset, batch_test_task_collate
import torch
from torch.utils.data import DataLoader

import torch
from torch.utils.data import DataLoader

# from dataloaders.loader import MyTestDataset, batch_test_task_collate
from dataloaders.loader_ova import MyDataset_OVA
from dataloaders.loader_ova import MyTestDataset, batch_test_task_collate
#from models.proto_learner import ProtoLearner
#from models.mpti_learner import MPTILearner
from utils.cuda_util import cast_cuda
from utils.logger import init_logger


def evaluate_metric(logger, pred_labels_list, gt_labels_list, label2class_list, test_classes):
    """
    :param pred_labels_list: a list of np array, each entry with shape (n_queries*n_way, num_points).
    :param gt_labels_list: a list of np array, each entry with shape (n_queries*n_way, num_points).
    :param test_classes: a list of np array, each entry with shape (n_way,)
    :return: iou: scaler
    """
    assert len(pred_labels_list) == len(gt_labels_list) == len(label2class_list)

    logger.cprint('*****Test Classes: {0}*****'.format(test_classes))

    NUM_CLASS = len(test_classes) + 1 # add 1 to consider background class
    # print('this is  the number of test_classes', NUM_CLASS)
    gt_classes = [0 for _ in range(NUM_CLASS)]
    positive_classes = [0 for _ in range(NUM_CLASS)]
    true_positive_classes = [0 for _ in range(NUM_CLASS)]

    for i, batch_gt_labels in enumerate(gt_labels_list):
        batch_pred_labels = pred_labels_list[i] #(n_queries*n_way, num_points)
        label2class = label2class_list[i] #(n_way,)
        # print('this is the bacth_pres_labels', batch_pred_labels)
        # print('this is the label2 class', label2class)        
        for j in range(batch_pred_labels.shape[0]):
            for k in range(batch_pred_labels.shape[1]):
                gt = int(batch_gt_labels[j, k])
                pred = int(batch_pred_labels[j,k])
                # print('this is the prediction inside the batch',pred)
                if gt == 0: # 0 indicate background class
                    gt_index = 0
                else:
                    gt_class = label2class[gt-1] # the ground truth class in the dataset
                    gt_index = test_classes.index(gt_class) + 1
                gt_classes[gt_index] += 1

                if pred == 0:
                    pred_index = 0
                else:
                    pred_class = label2class[pred-1]
                    pred_index = test_classes.index(pred_class) + 1
                    # print('This is the predicted class',pred_class)
                    
                positive_classes[pred_index] += 1

                true_positive_classes[gt_index] += int(gt == pred)

    iou_list = []
    for c in range(NUM_CLASS):
        iou = true_positive_classes[c] / float(gt_classes[c] + positive_classes[c] - true_positive_classes[c])
        logger.cprint('----- [class %d]  IoU: %f -----'% (c, iou))
        iou_list.append(iou)

    mean_IoU = np.array(iou_list[1:]).mean()

    return mean_IoU


def test_few_shot(test_loader, learner, logger, test_classes):

    total_loss = 0
    output_dir = '/home/farhan/attMPTI/output_vis'
    output_dir_scene_1 = '/home/farhan/attMPTI/scene_1'
    output_dir_scene_2 = '/home/farhan/attMPTI/scene_2'
    predicted_label_total = []
    gt_label_total = []
    label2class_total = []
    scene_1 =[]
    scene_2= []

    for batch_idx, (data, sampled_classes) in enumerate(test_loader):
        # query_label = data[-1]
        [support_x, support_y, query_x, query_label] = data
        # y_label=np.expand_dims(query_label[0,:], axis=0)
        # tmp=np.concatenate((query_x[0,:,:].numpy(),y_label),axis=0)
        #print("This is the shape of query:",query_label.shape)
        if torch.cuda.is_available():
            data = cast_cuda(data)
        # print('query label shape', query_label.shape)
        query_pred, loss, accuracy = learner.test(data)
        # print("Different Data Shape:", data[0].shape,data[1].shape,data[2].shape,data[3].shape)
        # print("DIff data shape:" , data[-1].shape,data[-2].shape,data[-3].shape,data[-4].shape)
        data_n = query_x.cpu().detach().numpy()
        # data_n = tmp
        total_loss += loss.detach().item()
        if (batch_idx+1) % 50 == 0:
            logger.cprint('[Eval] Iter: %d | Loss: %.4f | %s' % ( batch_idx+1, loss.detach().item(), str(datetime.now())))

        #compute metric for predictions
        predicted_label_total.append(query_pred.cpu().detach().numpy())
        gt_label_total.append(query_label.numpy())
        label2class_total.append(sampled_classes)
        
        # predicted_= (query_pred[0]).cpu().detach().numpy()
        # data = (data.cuda()).cpu()
        # print('this is the shape of data',data_n.shape)
        
        # # np.concatenate((data_n[1,:,:], predicted_label_total),axis=1)
        # # data_n[-1] = predicted_label_total[0]
        # print("this is new shape",data_n.shape)
        # print(predicted_label_total[0].shape)
        # print("Shape of predicted label", len(predicted_label_total))
        # print("label2class", label2class_total[0].shape)
        predicted_label_total_np = np.array(predicted_label_total)
        # print(query_label.shape)
        # print(predicted_label_total_np[.shape)
        # print(predicted_label_total_np.shape)
        # #save query 1:
        # print(data_n[0,:].shape)
        
        # print(predicted_label_total_np[0,0,:].shape)
        
        y_label=np.expand_dims(predicted_label_total_np[0,0,:], axis=0)
        # print(y_label.shape, data_n[0,:].shape)
        query_block_1 = np.concatenate((data_n[0,:],y_label),axis=0) 
        # print(query_block_1.shape)
        # print('THis is the real data',data_n[-2][1])
        scene_1.append(query_block_1.T)
        scene_1_np= np.array(scene_1)
        # print(query_block_1.shape,scene_1[0].shape)
        #save query 2:
        y_label_n=np.expand_dims(predicted_label_total_np[0,1,:], axis=0)
        query_block_2 = np.concatenate((data_n[1,:],y_label_n),axis=0) 
        # query_block_2 = np.concatenate((data_n[1,:,:],predicted_label_total_np[1,:]),axis=1) 
        scene_2.append(query_block_2.T)
        scene_2_np= np.array(scene_2)
        # print(query_block_2.shape,scene_2[0].shape)
        # data_n[-
        # np.concatenate((data_n, predicted_label_total),axis=1)
        # print(data_n.shape)
        np.save(os.path.join(output_dir_scene_1, f'{batch_idx}_sce_1_data.npy'), scene_1_np)
        np.save(os.path.join(output_dir_scene_2, f'{batch_idx}_sce_2_data.npy'), scene_2_np)
        #np.save(str(batch_idx)+'pred.npy', query_pred)
        #query_pred_n = query_pred.cpu().numpy()
        #np.save(str(batch_idx)+'predited.npy',query_pred)
        #np.save(str(batch_idx)+'data.npy', data)
        
    print('Length of the scene', scene_1_np.shape, scene_2_np.shape)   
   
    
    # print('Shape of the predicted label array is', len(predicted_label_total))
    # print("===========================================+++++++++++================= This is Predicted LABEL+++++++")
    # print(np.unique(predicted_label_total))
    
    mean_loss = total_loss/len(test_loader)
    mean_IoU = evaluate_metric(logger, predicted_label_total, gt_label_total, label2class_total, test_classes)
    return mean_loss, mean_IoU


def eval(args):
    logger = init_logger(args.log_dir, args)

    if args.phase == 'protoeval':
        learner = ProtoLearner(args, mode='test')
    elif args.phase == 'mptieval':
        learner = MPTILearner(args, mode='test')

    #Init dataset, dataloader
    # TEST_DATASET = MyTestDataset(args.data_path, args.dataset, cvfold=args.cvfold,
    #                              num_episode_per_comb=args.n_episode_test,
    #                              n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
    #                              num_point=args.pc_npts, pc_attribs=args.pc_attribs,  mode='test')
    TEST_DATASET = MyTestDataset(args.example, args.data_path, args.ova_data_path, args.dataset,cvfold=args.cvfold,
                                num_episode_per_comb=args.n_episode_test,
                                n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,num_point=args.pc_npts, pc_attribs=args.pc_attribs,  mode='test')
    TEST_CLASSES = list(TEST_DATASET.classes)
    TEST_LOADER = DataLoader(TEST_DATASET, batch_size=1, shuffle=False, collate_fn=batch_test_task_collate)

    test_loss, mean_IoU = test_few_shot(TEST_LOADER, learner, logger, TEST_CLASSES)

    logger.cprint('\n=====[TEST] Loss: %.4f | Mean IoU: %f =====\n' %(test_loss, mean_IoU))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--phase', type=str, default='mptitrain', choices=['pretrain', 'finetune',
                                                                           'prototrain', 'protoeval',
                                                                           'mptitrain', 'mptieval'])
    parser.add_argument('--example', type=int, default=0,
                        help='if we have example in OVA')
    parser.add_argument('--dataset', type=str, default='s3dis', help='Dataset name: s3dis|scannet')
    parser.add_argument('--cvfold', default=['chair','table'], help='DGCNN Edgeconv widths')
    parser.add_argument('--data_path', type=str, default='../blocks_bs1_s1',
                        help='Directory to the source data')
    parser.add_argument('--ova_data_path', type=str, default='../OVA_Dataset/',
                        help='Directory to the source data')
    parser.add_argument('--ova_example', default=False)
    parser.add_argument('--pretrain_checkpoint_path', type=str, default=None,
                        help='Path to the checkpoint of pre model for resuming')
    parser.add_argument('--model_checkpoint_path', type=str, default=None,
                        help='Path to the checkpoint of model for resuming')
    parser.add_argument('--save_path', type=str, default='./log_s3dis/',
                        help='Directory to the save log and checkpoints')
    parser.add_argument('--eval_interval', type=int, default=1500,
                        help='iteration/epoch inverval to evaluate model')

    # optimization
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples/tasks in one batch')
    parser.add_argument('--n_workers', type=int, default=16, help='number of workers to load data')
    parser.add_argument('--n_iters', type=int, default=30000, help='number of iterations/epochs to train')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Model (eg. protoNet or MPTI) learning rate [default: 0.001]')
    parser.add_argument('--step_size', type=int, default=5000, help='Iterations of learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')

    parser.add_argument('--pretrain_lr', type=float, default=0.001, help='pretrain learning rate [default: 0.001]')
    parser.add_argument('--pretrain_weight_decay', type=float, default=0., help='weight decay for regularization')
    parser.add_argument('--pretrain_step_size', type=int, default=50, help='Period of learning rate decay')
    parser.add_argument('--pretrain_gamma', type=float, default=0.5,
                        help='Multiplicative factor of learning rate decay')

    # few-shot episode setting
    parser.add_argument('--n_way', type=int, default=2, help='Number of classes for each episode: 1|3')
    parser.add_argument('--k_shot', type=int, default=1, help='Number of samples/shots for each class: 1|5')
    parser.add_argument('--n_queries', type=int, default=1, help='Number of queries for each class')
    parser.add_argument('--n_episode_test', type=int, default=100,
                        help='Number of episode per configuration during testing')

    # Point cloud processing
    parser.add_argument('--pc_npts', type=int, default=2048, help='Number of input points for PointNet.')
    parser.add_argument('--pc_attribs', default='xyzrgbXYZ',
                        help='Point attributes fed to PointNets, if empty then all possible. '
                             'xyz = coordinates, rgb = color, XYZ = normalized xyz')
    parser.add_argument('--pc_augm', action='store_true', help='Training augmentation for points in each superpoint')
    parser.add_argument('--pc_augm_scale', type=float, default=0,
                        help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', type=int, default=1,
                        help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', type=float, default=0,
                        help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', type=int, default=1,
                        help='Training augmentation: Bool, Gaussian jittering of all attributes')

    # feature extraction network configuration
    parser.add_argument('--dgcnn_k', type=int, default=20, help='Number of nearest neighbors in Edgeconv')
    parser.add_argument('--edgeconv_widths', default='[[64,64], [64,64], [64,64]]', help='DGCNN Edgeconv widths')
    parser.add_argument('--dgcnn_mlp_widths', default='[512, 256]',
                        help='DGCNN MLP (following stacked Edgeconv) widths')
    parser.add_argument('--base_widths', default='[128, 64]', help='BaseLearner widths')
    parser.add_argument('--output_dim', type=int, default=64,
                        help='The dimension of the final output of attention learner or linear mapper')
    parser.add_argument('--use_attention', action='store_true', help='if incorporate attention learner')

    # protoNet configuration
    parser.add_argument('--dist_method', default='euclidean',
                        help='Method to compute distance between query feature maps and prototypes.[Option: cosine|euclidean]')

    # MPTI configuration
    parser.add_argument('--n_subprototypes', type=int, default=100,
                        help='Number of prototypes for each class in support set')
    parser.add_argument('--k_connect', type=int, default=200,
                        help='Number of nearest neighbors to construct local-constrained affinity matrix')
    parser.add_argument('--sigma', type=float, default=1., help='hyeprparameter in gaussian similarity function')

    args = parser.parse_args()

    args.edgeconv_widths = ast.literal_eval(args.edgeconv_widths)
    args.dgcnn_mlp_widths = ast.literal_eval(args.dgcnn_mlp_widths)
    args.base_widths = ast.literal_eval(args.base_widths)
    args.pc_in_dim = len(args.pc_attribs)
    PC_AUGMENT_CONFIG = {'scale': args.pc_augm_scale,
                         'rot': args.pc_augm_rot,
                         'mirror_prob': args.pc_augm_mirror_prob,
                         'jitter': args.pc_augm_jitter
                         }
    TEST_DATASET = MyTestDataset(args.example, args.data_path, args.ova_data_path, args.dataset, cvfold=args.cvfold,
                                 num_episode_per_comb=args.n_episode_test,
                                 n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries, num_point=args.pc_npts,
                                 pc_attribs=args.pc_attribs, mode='test')
    TEST_CLASSES = list(TEST_DATASET.classes)
    test_loader = DataLoader(TEST_DATASET, batch_size=1, shuffle=False, collate_fn=batch_test_task_collate)
    query_scene =[]
    for batch_idx, (data, sampled_classes,xyz_min) in enumerate(test_loader):
        [support_x, support_y, query_x, query_label] = data
        if torch.cuda.is_available():
            data = cast_cuda(data)
        # print('query label shape', query_label.shape)
        #query_pred, loss, accuracy = learner.test(data)
        """
        query_pred = query_pred.cpu().detach().numpy()
        query_point = query_x.cpu().detach().numpy()

        for q in range(query_point.shape[0]):
            y_label = np.expand_dims(query_pred[q, :], axis=0)
            #y_label = query_pred[q, :]
            query_block = np.concatenate((query_point[q, :,:], y_label), axis=0)
            query_scene.append(query_block.T)

    query_scene = np.vstack(query_scene)
    """
        query_point = query_x.cpu().detach().numpy()
        for q in range(query_point.shape[0]):
            query_scene.append(query_point[q,0:3,:].T+xyz_min[q])
    query_scene = np.vstack(query_scene)
    np.save(os.path.join('../Output','Scene.npy'), query_scene)

    print('Length of the scene', query_scene.shape)

