import open3d as o3d
import numpy as np


mesh_pred = np.load('Output/Scene.npy')
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(mesh_pred[:,:3])
#pcd.colors = o3d.utility.Vector3dVector(mesh_input[:,3:6])
o3d.visualization.draw_geometries([pcd])