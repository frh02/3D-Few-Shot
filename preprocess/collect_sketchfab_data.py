""" Collect point clouds and the corresponding labels from original ScanNetV2 dataset, and save into numpy files.

Author: Zhao Na, 2020
"""
import os
import sys
import json
import numpy as np
from plyfile import PlyData

def read_ply_xyzrgb(filename):
    """ read XYZRGB point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
   
    return vertices

def collect_point_label(ply_filename, out_filename):
    # if label_file:
    #     pass
    # else:
    labels= ['unannotated','table','chair','tv','floor','doorframe','door','object','ceiling','kitchen cabinets','cabinets','bed','regrigerator']
    # label= 'unannotated'
    instance_points_list= []
    semantic_labels_list = []
    # scene_name = 'col_tavern'
    points = read_ply_xyzrgb(ply_filename)
    # print('{0}: {1} points'.format(points.shape[0]))

    # ply_filename = os.path.join(scene_path, '%s_vh_clean_2.ply' % (scene_name))
    points = read_ply_xyzrgb(ply_filename)
    # print('{0}: {1} points'.format(scene_name, points.shape[0]))
    instance_points = points[:]
    instance_points_list.append(instance_points)
    print(instance_points_list[0].shape)
    label = 'object'
    label = labels.index(label)
    print(label)
    semantic_labels_list.append(np.ones((instance_points.shape[0], 1)) * label)
    
    scene_points = np.concatenate(instance_points_list, 0)
    scene_points = scene_points[:, 0:6]  # XYZRGB, disregarding the A
    # instance_labels = np.concatenate(instance_labels_list, 0)
    print(scene_points.shape)
    print(label)
    print(instance_points.shape[0])
    print('hi',semantic_labels_list[0].shape)
    
    # semantic_labels_list.append(np.ones((instance_points.shape[0], 1)) * label)
    print(semantic_labels_list[0].shape)
    semantic_labels = np.concatenate(semantic_labels_list, 0)
    print(semantic_labels.shape)
    
    # data = np.concatenate((scene_points, instance_labels, semantic_labels), 1)
    data = np.concatenate((scene_points, semantic_labels), 1)
    print(data.shape)
    print(np.unique(label))
    np.save(out_filename, data)
   


if __name__ == '__main__':
    # import argparse
    scene_name = 'col_tavern'
    out_filename = 'colored_tavern.npy'
    filename= '/home/farhan/attMPTI/scene_col.ply'
    # read_ply_xyzrgb(filename)
    collect_point_label(filename, out_filename)

