import open3d as o3d
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import numpy as np
import torch
from pcdet_utils import calibration_kitti
import matplotlib.pyplot as plt
import copy

class Painter:
    def __init__(self):
        self.root_split_path = "../LidarDetector/data/kitti/training/"
        self.save_path = "../LidarDetector/data/kitti/training/painted_lidar_hma/"

    def get_lidar(self, idx):
        lidar_file = self.root_split_path + 'velodyne/' + ('%s.bin' % idx)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_score(self, idx, left):
        filename = self.root_split_path + left + ('%s.npy' % idx)
        output_reassign_softmax = np.load(filename)
        return output_reassign_softmax

    def get_label(self, idx):
        label_file = self.root_split_path + 'label_2/' + ('%s.txt' % idx)
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path + 'calib/' + ('%s.txt' % idx)
        return calibration_kitti.Calibration(calib_file)
    
    def get_calib_fromfile(self, idx):
        calib_file = self.root_split_path + 'calib/' + ('%s.txt' % idx)
        calib = calibration_kitti.get_calib_from_file(calib_file)
        calib['P2'] = np.concatenate([calib['P2'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['P3'] = np.concatenate([calib['P3'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['R0_rect'] = np.zeros([4, 4], dtype=calib['R0'].dtype)
        calib['R0_rect'][3, 3] = 1.
        calib['R0_rect'][:3, :3] = calib['R0']
        calib['Tr_velo2cam'] = np.concatenate([calib['Tr_velo2cam'], np.array([[0., 0., 0., 1.]])], axis=0)
        return calib
    
    def create_cyclist(self, augmented_lidar):
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

    def cam_to_lidar(self, pointcloud, projection_mats):
        """
        Takes in lidar in velo coords, returns lidar points in camera coords

        :param pointcloud: (n_points, 4) np.array (x,y,z,r) in velodyne coordinates
        :return lidar_cam_coords: (n_points, 4) np.array (x,y,z,r) in camera coordinates
        """

        lidar_velo_coords = copy.deepcopy(pointcloud)
        reflectances = copy.deepcopy(lidar_velo_coords[:, -1]) #copy reflectances column
        lidar_velo_coords[:, -1] = 1 # for multiplying with homogeneous matrix
        lidar_cam_coords = projection_mats['Tr_velo2cam'].dot(lidar_velo_coords.transpose())
        lidar_cam_coords = lidar_cam_coords.transpose()
        lidar_cam_coords[:, -1] = reflectances
        
        return lidar_cam_coords

    def augment_lidar_class_scores(self, class_scores, lidar_raw, projection_mats):
        """
        Projects lidar points onto segmentation map, appends class score each point projects onto.
        """
        lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        points_projected_on_mask = projection_mats['P2'].dot(projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask = points_projected_on_mask.transpose()
        points_projected_on_mask = points_projected_on_mask/(points_projected_on_mask[:,2].reshape(-1,1))

        true_where_x_on_img = (0 < points_projected_on_mask[:, 0]) & (points_projected_on_mask[:, 0] < class_scores.shape[1]) #x in img coords is cols of img
        true_where_y_on_img = (0 < points_projected_on_mask[:, 1]) & (points_projected_on_mask[:, 1] < class_scores.shape[0])
        true_where_point_on_img = true_where_x_on_img & true_where_y_on_img

        points_projected_on_mask = points_projected_on_mask[true_where_point_on_img] # filter out points that don't project to image
        lidar_cam_coords = lidar_cam_coords[true_where_point_on_img]
        points_projected_on_mask = np.floor(points_projected_on_mask).astype(int) # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask = points_projected_on_mask[:, :2] #drops homogenous coord 1 from every point, giving (N_pts, 2) int array
        
        #indexing oreder below is 1 then 0 because points_projected_on_mask is x,y in image coords which is cols, rows while class_score shape is (rows, cols)
        #socre dimesion: point_scores.shape[2] TODO!!!!
        point_scores = class_scores[points_projected_on_mask[:, 1], points_projected_on_mask[:, 0]].reshape(-1, class_scores.shape[2])
        augmented_lidar = np.concatenate((lidar_raw[true_where_point_on_img], point_scores), axis=1)
        augmented_lidar = self.create_cyclist(augmented_lidar)

        return augmented_lidar

    def augment_lidar_class_scores_both(self, class_scores_r, class_scores_l, lidar_raw, projection_mats):
        """
        Projects lidar points onto segmentation map, appends class score each point projects onto.
        """
        #lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)
        # TODO: Project lidar points onto left and right segmentation maps. How to use projection_mats? 
        ################################
        lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)

        # right
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_r = projection_mats['P3'].dot(projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask_r = points_projected_on_mask_r.transpose()
        points_projected_on_mask_r = points_projected_on_mask_r/(points_projected_on_mask_r[:,2].reshape(-1,1))

        true_where_x_on_img_r = (0 < points_projected_on_mask_r[:, 0]) & (points_projected_on_mask_r[:, 0] < class_scores_r.shape[1]) #x in img coords is cols of img
        true_where_y_on_img_r = (0 < points_projected_on_mask_r[:, 1]) & (points_projected_on_mask_r[:, 1] < class_scores_r.shape[0])
        true_where_point_on_img_r = true_where_x_on_img_r & true_where_y_on_img_r

        points_projected_on_mask_r = points_projected_on_mask_r[true_where_point_on_img_r] # filter out points that don't project to image
        points_projected_on_mask_r = np.floor(points_projected_on_mask_r).astype(int) # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask_r = points_projected_on_mask_r[:, :2] #drops homogenous coord 1 from every point, giving (N_pts, 2) int array

        # left
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_l = projection_mats['P2'].dot(projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask_l = points_projected_on_mask_l.transpose()
        points_projected_on_mask_l = points_projected_on_mask_l/(points_projected_on_mask_l[:,2].reshape(-1,1))

        true_where_x_on_img_l = (0 < points_projected_on_mask_l[:, 0]) & (points_projected_on_mask_l[:, 0] < class_scores_l.shape[1]) #x in img coords is cols of img
        true_where_y_on_img_l = (0 < points_projected_on_mask_l[:, 1]) & (points_projected_on_mask_l[:, 1] < class_scores_l.shape[0])
        true_where_point_on_img_l = true_where_x_on_img_l & true_where_y_on_img_l

        points_projected_on_mask_l = points_projected_on_mask_l[true_where_point_on_img_l] # filter out points that don't project to image
        points_projected_on_mask_l = np.floor(points_projected_on_mask_l).astype(int) # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask_l = points_projected_on_mask_l[:, :2] #drops homogenous coord 1 from every point, giving (N_pts, 2) int array

        true_where_point_on_both_img = true_where_point_on_img_l & true_where_point_on_img_r
        true_where_point_on_img = true_where_point_on_img_l | true_where_point_on_img_r

        #indexing oreder below is 1 then 0 because points_projected_on_mask is x,y in image coords which is cols, rows while class_score shape is (rows, cols)
        #socre dimesion: point_scores.shape[2] TODO!!!!
        point_scores_r = class_scores_r[points_projected_on_mask_r[:, 1], points_projected_on_mask_r[:, 0]].reshape(-1, class_scores_r.shape[2])
        point_scores_l = class_scores_l[points_projected_on_mask_l[:, 1], points_projected_on_mask_l[:, 0]].reshape(-1, class_scores_l.shape[2])
        #augmented_lidar = np.concatenate((lidar_raw[true_where_point_on_img], point_scores), axis=1)
        augmented_lidar = np.concatenate((lidar_raw, np.zeros((lidar_raw.shape[0], class_scores_r.shape[2]))), axis=1)
        augmented_lidar[true_where_point_on_img_r, -class_scores_r.shape[2]:] += point_scores_r
        augmented_lidar[true_where_point_on_img_l, -class_scores_l.shape[2]:] += point_scores_l
        augmented_lidar[true_where_point_on_both_img, -class_scores_r.shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_img, -class_scores_r.shape[2]:]
        augmented_lidar = augmented_lidar[true_where_point_on_img]
        augmented_lidar = self.create_cyclist(augmented_lidar)

        return augmented_lidar

    def run(self):
        num_image = 7481
        for idx in range(num_image):
            print(idx)
            idx=100
            sample_idx = "%06d" % idx
            points = self.get_lidar(sample_idx)
            # print(points.shape)
            # add socres here
            scores_from_cam = self.get_score(sample_idx, "score_2_hma/")
            scores_from_cam_r = self.get_score(sample_idx, "score_3_hma/")
            # points become [points, scores]
            calib_fromfile = self.get_calib_fromfile(sample_idx)
            
            # points = self.augment_lidar_class_scores(scores_from_cam, points, calib_info)
            points = self.augment_lidar_class_scores_both(scores_from_cam_r, scores_from_cam, points, calib_fromfile)

            # print(points.shape)
            # open3d
            point_cloud = o3d.geometry.PointCloud()
            np_points = np.array(points[:,:3]) #xyz
            np_color = np.array(points[:,5:8])
            point_cloud.points = o3d.utility.Vector3dVector(np_points)
            point_cloud.colors = o3d.utility.Vector3dVector(np_color)
            o3d.visualization.draw_geometries([point_cloud])

            np.save(self.save_path + ("%06d.npy" % idx), points)

if __name__ == '__main__':
    painter = Painter()
    painter.run()
