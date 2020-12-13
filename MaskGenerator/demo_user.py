#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
from open3d import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np
from tasks.semantic.modules.segmentator import *
from tasks.semantic.modules.trainer import *
from tasks.semantic.postproc.KNN import KNN

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

class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=None,
                                      valid_sequences=None,
                                      test_sequences=self.DATA["split"]["sample"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = Segmentator(self.ARCH,
                               self.parser.get_n_classes(),
                               self.modeldir)

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):
    # do test set
    self.infer_subset(loader=self.parser.get_test_set(),
                      to_orig_fn=self.parser.to_original)

    print('Finished Infering')

    return

  def infer_subset(self, loader, to_orig_fn):
    # switch to evaluate mode
    self.model.eval()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()

      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, unproj_xyz, _, unproj_remissions, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          proj_mask = proj_mask.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()

          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        proj_output, _, _, _, _ = self.model(proj_in, proj_mask)
        print(proj_output.shape)
        print(sum(proj_output[0,:,0,0]))
        print((proj_output[0,:,0,0]).argmax(dim=0))
        proj_argmax = proj_output[0].argmax(dim=0)
        print(proj_argmax.shape)

        # if self.post:
        #   # knn postproc
        #   unproj_argmax = self.post(proj_range,
        #                             unproj_range,
        #                             proj_argmax,
        #                             p_x,
        #                             p_y)
        # else:
        #   # put in original pointcloud using indexes
        #   unproj_argmax = proj_argmax[p_y, p_x]

        print(p_y)
        print(p_x)
        print(proj_argmax[p_y, p_x].shape)
        unproj_output = proj_output[0,:,p_y,p_x]
        print(unproj_output.shape)
        print(unproj_output[:,0])

        output_permute = unproj_output.permute(1,0)
        print(output_permute.shape)

        output_reassign = torch.zeros(output_permute.size(0), 5)
        output_reassign[:,0], _ = torch.max(output_permute[:,[0,9,10,11,12,13,14,15,16,17,18,19]], dim=1) # background
        output_reassign[:,1], _ = torch.max(output_permute[:,[2, 3]], dim=1) # bicycle
        output_reassign[:,2], _ = torch.max(output_permute[:,[1, 4, 5]], dim=1) # car
        output_reassign[:,3] = output_permute[:,6] #person
        output_reassign[:,4], _ = torch.max(output_permute[:,[7, 8]], dim=1) #rider
        print(output_reassign[0,:])
        sf = torch.nn.Softmax(dim=1)
        output_reassign_softmax = sf(output_reassign)       
        print(sum(output_reassign_softmax[0,:]))
        print(unproj_remissions.shape)
        points = (torch.cat((unproj_xyz[0][:npoints,:],unproj_remissions[0][:npoints].reshape(-1,1), output_reassign_softmax), dim=1)).detach().numpy()
        points = create_cyclist(points)

        # open3d
        # point_cloud = PointCloud()
        # np_points = np.array(points[:,:3]) #xyz
        # np_color = np.array(points[:,5:8])
        # point_cloud.points = Vector3dVector(np_points)
        # point_cloud.colors = Vector3dVector(np_color)
        # draw_geometries([point_cloud])
        save_file_name = path_name.replace('.label', '.npy')
        print(save_file_name)
        output_dir = '/home/lethe5lambda/Documents/PointPainter/LidarDetector/data/kitti/training/painted_lidar_sqseg/'
        np.save(output_dir + save_file_name, points)
        
        if torch.cuda.is_available():
         torch.cuda.synchronize()

        # print("Infered seq", path_seq, "scan", path_name,
        #       "in", time.time() - end, "sec")
        
        # end = time.time()

        # # save scan
        # # get the first scan in batch and project scan
        # pred_np = unproj_argmax.cpu().numpy()
        # pred_np = pred_np.reshape((-1)).astype(np.int32)

        # # map to original label
        # pred_np = to_orig_fn(pred_np)

        # # save scan
        # path = os.path.join(self.logdir, "sequences",
        #                     path_seq, "predictions", path_name)
        # print(pred_np)
        # pred_np.tofile(path)
        # depth = (cv2.normalize(proj_in[0][0].cpu().numpy(), None, alpha=0, beta=1,
        #                    norm_type=cv2.NORM_MINMAX,
        #                    dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        # print(depth.shape, proj_mask.shape,proj_argmax.shape)
        # out_img = cv2.applyColorMap(
        #     depth, Trainer.get_mpl_colormap('viridis')) * proj_mask[0].cpu().numpy()[..., None]
        #  # make label prediction
        # pred_color = self.parser.to_color((proj_argmax.cpu().numpy() * proj_mask[0].cpu().numpy()).astype(np.int32))
        # out_img = np.concatenate([out_img, pred_color], axis=0)
        # print(path)
        # cv2.imwrite(path[:-6]+'.png',out_img)


