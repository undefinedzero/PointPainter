''' Visualization code for point clouds and 3D bounding boxes with mayavi.

Modified by Charles R. Qi
Date: September 2017

Ref: https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/kitti_data/draw.py

Modified by Haotian Tang
Date: August 2020
'''

import argparse
import numpy as np
import mayavi.mlab as mlab
import os
from open3d import *
import torch
from torchsparse.utils import sparse_quantize, sparse_collate
from torchsparse import SparseTensor
from model_zoo import minkunet, spvcnn, spvnas_specialized


def process_point_cloud(input_point_cloud, input_labels=None, voxel_size=0.05):
    input_point_cloud[:, 3] = input_point_cloud[:, 3]
    pc_ = np.round(input_point_cloud[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)
    
    label_map = create_label_map()
    if input_labels is not None:
        labels_ = label_map[input_labels & 0xFFFF].astype(
            np.int64)  # semantic labels
    else:
        labels_ = np.zeros(pc_.shape[0], dtype=np.int64)
    
    feat_ = input_point_cloud
    
    if input_labels is not None:
        out_pc = input_point_cloud[labels_ != labels_.max(), :3]
        pc_ = pc_[labels_ != labels_.max()]
        feat_ = feat_[labels_ != labels_.max()]
        labels_ = labels_[labels_ != labels_.max()]
    else:
        out_pc = input_point_cloud
        pc_ = pc_
        
    inds, labels, inverse_map = sparse_quantize(pc_,
                                                feat_,
                                                labels_,
                                                return_index=True,
                                                return_invs=True)
    pc = np.zeros((inds.shape[0], 4))
    pc[:, :3] = pc_[inds]
    
    feat = feat_[inds]
    labels = labels_[inds]
    lidar = SparseTensor(
        torch.from_numpy(feat).float(), 
        torch.from_numpy(pc).int()
    )
    return {
        'pc': out_pc,
        'lidar': lidar,
        'targets': labels,
        'targets_mapped': labels_,
        'inverse_map': inverse_map
    }


mlab.options.offscreen = True

def create_label_map(num_classes=19):
    name_label_mapping = {
        'unlabeled': 0, 'outlier': 1, 'car': 10, 'bicycle': 11,
        'bus': 13, 'motorcycle': 15, 'on-rails': 16, 'truck': 18,
        'other-vehicle': 20, 'person': 30, 'bicyclist': 31,
        'motorcyclist': 32, 'road': 40, 'parking': 44,
        'sidewalk': 48, 'other-ground': 49, 'building': 50,
        'fence': 51, 'other-structure': 52, 'lane-marking': 60,
        'vegetation': 70, 'trunk': 71, 'terrain': 72, 'pole': 80,
        'traffic-sign': 81, 'other-object': 99, 'moving-car': 252,
        'moving-bicyclist': 253, 'moving-person': 254, 'moving-motorcyclist': 255,
        'moving-on-rails': 256, 'moving-bus': 257, 'moving-truck': 258,
        'moving-other-vehicle': 259
    }
    
    for k in name_label_mapping:
        name_label_mapping[k] = name_label_mapping[k.replace('moving-', '')]
    train_label_name_mapping = {
        0: 'car', 1: 'bicycle', 2: 'motorcycle', 3: 'truck', 4:
        'other-vehicle', 5: 'person', 6: 'bicyclist', 7: 'motorcyclist',
        8: 'road', 9: 'parking', 10: 'sidewalk', 11: 'other-ground',
        12: 'building', 13: 'fence', 14: 'vegetation', 15: 'trunk',
        16: 'terrain', 17: 'pole', 18: 'traffic-sign'
    }

    label_map = np.zeros(260)+num_classes
    for i in range(num_classes):
        cls_name = train_label_name_mapping[i]
        label_map[name_label_mapping[cls_name]] = min(num_classes,i)
    return label_map.astype(np.int64)

cmap = np.array([
    [245, 150, 100, 255],#[0,0,0] for teaser1
    [245, 230, 100, 255],
    [150, 60, 30, 255],
    [180, 30, 80, 255],
    [255, 0, 0, 255],
    [30, 30, 255, 255],#[255,0,0] for teaser1
    [200, 40, 255, 255],
    [90, 30, 150, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [75, 0, 175, 255],
    [0, 200, 255, 255],
    [50, 120, 255, 255],
    [0, 175, 0, 255],
    [0, 60, 135, 255],
    [80, 240, 150, 255],
    [150, 240, 255, 255],
    [0, 0, 255, 255],
    #[255, 255, 255, 0]
])


#cmap[:, -1] = 255
def create_cyclist(augmented_lidar):
    rider_idx = np.where(augmented_lidar[:,8]>=0.3)[0] # 0, 1(bike), 2, 3(person), 4(rider)
    rider_points = augmented_lidar[rider_idx]
    bike_mask_total = np.zeros(augmented_lidar.shape[0], dtype=bool)
    bike_total = (np.argmax(augmented_lidar[:,-5:],axis=1) == 1)
    for i in range(rider_idx.shape[0]):
        bike_mask = (np.linalg.norm(augmented_lidar[:,:3]-rider_points[i,:3], axis=1) < 1) & bike_total
        bike_mask_total |= bike_mask
    augmented_lidar[bike_mask_total, 8] = augmented_lidar[bike_mask_total, 5]
    augmented_lidar[bike_total^bike_mask_total, 4] = augmented_lidar[bike_total^bike_mask_total, 5]
    return augmented_lidar[:,[0,1,2,3,4,8,6,7]]


def draw_lidar(pc, color=None, fig=None, bgcolor=(1,1,1), pts_scale=0.06, pts_mode='2dcircle', pts_color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(800, 500))
    if color is None: color = pc[:,2]
    pts = mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, mode=pts_mode, scale_factor=pts_scale, figure=fig)
    pts.glyph.scale_mode = 'scale_by_vector'
    pts.glyph.color_mode = 'color_by_scalar' # Color by scalar
    pts.module_manager.scalar_lut_manager.lut.table = cmap
    pts.module_manager.scalar_lut_manager.lut.number_of_colors = cmap.shape[0]
    
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62, figure=fig)
    
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--velodyne-dir', type=str, default='/home/lethe5lambda/Documents/PointPainter/LidarDetector/data/kitti/training/velodyne/')
    parser.add_argument('--model', type=str, default='SemanticKITTI_val_SPVCNN@119GMACs')
    args = parser.parse_args()
    output_dir = "/home/lethe5lambda/Documents/PointPainter/LidarDetector/data/kitti/training/painted_lidar_e3d/"
    os.makedirs(output_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    if 'MinkUNet' in args.model:
        model = minkunet(args.model, pretrained=True)
    elif 'SPVCNN' in args.model:
        model = spvcnn(args.model, pretrained=True)
    elif 'SPVNAS' in args.model:
        model = spvnas_specialized(args.model, pretrained=True)
    else:
        raise NotImplementedError
    
    model = model.to(device)
    
    input_point_clouds = sorted(os.listdir(args.velodyne_dir))
    for point_cloud_name in input_point_clouds:
        if not point_cloud_name.endswith('.bin'):
            continue
        label_file_name = point_cloud_name.replace('.bin', '.label')
        vis_file_name = point_cloud_name.replace('.bin', '.png')
        gt_file_name = point_cloud_name.replace('.bin', '_GT.png')
        save_file_name = point_cloud_name.replace('.bin', '.npy')
        
        pc = np.fromfile('%s/%s'%(args.velodyne_dir, point_cloud_name), dtype=np.float32).reshape(-1,4)
        if os.path.exists(label_file_name):
            label = np.fromfile('%s/%s'%(args.velodyne_dir, label_file_name), dtype=np.int32)
        else:
            label = None
        feed_dict = process_point_cloud(pc, label)
        inputs = feed_dict['lidar'].to(device)
        outputs = model(inputs)
        predictions = cmap[outputs.argmax(1).cpu().numpy()]/255.0
        predictions = predictions[feed_dict['inverse_map']]
        # predictions = np.concatenate((predictions,predictions,predictions),axis=1)
        print(save_file_name)
        output_permute = outputs[feed_dict['inverse_map']]
        # train_label_name_mapping = {
        #     0: 'car', 1: 'bicycle', 2: 'motorcycle', 3: 'truck', 4:
        #     'other-vehicle', 5: 'person', 6: 'bicyclist', 7: 'motorcyclist',
        #     8: 'road', 9: 'parking', 10: 'sidewalk', 11: 'other-ground',
        #     12: 'building', 13: 'fence', 14: 'vegetation', 15: 'trunk',
        #     16: 'terrain', 17: 'pole', 18: 'traffic-sign'
        # }
        sf = torch.nn.Softmax(dim=1)

        output_reassign = torch.zeros(output_permute.size(0), 5)
        output_reassign[:,0], _ = torch.max(output_permute[:,8:], dim=1) # background
        output_reassign[:,1], _ = torch.max(output_permute[:,[1, 2]], dim=1) # bicycle
        output_reassign[:,2], _ = torch.max(output_permute[:,[0, 3, 4]], dim=1) # car
        output_reassign[:,3] = output_permute[:,5] #person
        output_reassign[:,4], _ = torch.max(output_permute[:,[6, 7]], dim=1) #rider
        output_reassign_softmax = sf(output_reassign)
        points = (torch.cat((torch.tensor(feed_dict['pc']), output_reassign_softmax), dim=1)).detach().numpy()

        
        
        points = create_cyclist(points)

        # open3d
        point_cloud = PointCloud()
        np_points = np.array(points[:,:3]) #xyz
        np_color = np.array(points[:,5:8])
        point_cloud.points = Vector3dVector(np_points)
        point_cloud.colors = Vector3dVector(np_color)
        draw_geometries([point_cloud])

        np.save(output_dir + save_file_name, points)
