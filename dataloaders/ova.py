""" Data Preprocess and Loader for ScanNetV2 Dataset

Author: Zhao Na, 2020
"""
import os
import glob
import numpy as np
import pickle


class OvaDataset(object):
    def __init__(self,cvfold,dataset_name, ova_data_path):
        self.data_path = ova_data_path
        self.classes = 2
        # self.class2type = {0:'unannotated', 1:'wall', 2:'floor', 3:'chair', 4:'table', 5:'desk', 6:'bed', 7:'bookshelf',
        #                    8:'sofa', 9:'sink', 10:'bathtub', 11:'toilet', 12:'curtain', 13:'counter', 14:'door',
        #                    15:'window', 16:'shower curtain', 17:'refridgerator', 18:'picture', 19:'cabinet', 20:'otherfurniture'}
        if dataset_name =='s3dis':
            class_names = open(
                os.path.join('../meta', 's3dis_classnames.txt')).readlines()
        else:
            class_names = open('../meta/scannet_classnames.txt').readlines()
        self.class2type = {i: name.strip() for i, name in enumerate(class_names)}
        self.type2class = {self.class2type[t]: t for t in self.class2type}
        self.types = self.type2class.keys()
        self.test_classes = [self.type2class[i] for i in cvfold]




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pre-training on ShapeNet')
   
    parser.add_argument('--data_path', type=str, default='/home/farhan/attMPTI/datasets/SketchFab/blocks_bs1_s1/', help='Directory to source data')
    args = parser.parse_args()
    dataset = OvaDataset(args.data_path)