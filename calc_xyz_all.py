import numpy as np
import cv2
import math
from os.path import join

class camera_realtimeXYZ:
    #filenames for cam matrix, distortion, and new cam matrix
    paths = {0:('cameraMatrix.npy', 'camera_left_249cameraDistortion.npy','camera_left_249newcam_mtx.npy'),
            1: ('cameraMatrix.npy', 'camera_right_250cameraDistortion.npy','camera_right_250newcam_mtx.npy')}
    #directory for left and right camera configs
    savedir = {0:'camera_data1', 1: "camera_data2"}
    
    def __init__(self, i=0):
        self.cam_mtx = np.load(join(camera_realtimeXYZ.savedir[i], camera_realtimeXYZ.paths[i][0]))
        self.dist = np.load(join(camera_realtimeXYZ.savedir[i], camera_realtimeXYZ.paths[i][1]))
        self.newcam_mtx = np.load(join(camera_realtimeXYZ.savedir[i], camera_realtimeXYZ.paths[i][2]))
       # self.roi=np.load(savedir+'roi.npy')        
        #self.R_mtx=R_mtx
       # self.Rt=np.load(savedir+'Rt.npy')
       # self.P_mtx=np.load(savedir+'P_mtx.npy')

       # s_arr=np.load(savedir+'s_arr.npy')
       # self.scalingfactor=s_arr[0]

        self.inverse_newcam_mtx = np.linalg.inv(self.newcam_mtx)
        #self.inverse_R_mtx = np.linalg.inv(self.R_mtx)

    def undistort_image(self,image):
        image_undst = cv2.undistort(image, self.cam_mtx, self.dist, None, self.newcam_mtx)

        return image_undst

    def camera_center(self):
        return [self.cam_mtx[0][2], self.cam_mtx[1][2]]

    def new_camera_center(self):
        return [self.newcam_mtx[0][2], self.newcam_mtx[1][2]]

    def calculate_XYZ(self,u,v, R_mtx, R_mtx_cam, return_cam = False):
                                      
        #Solve: From Image Pixels, find World Points

        uv_1=np.array([[u,v,1]], dtype=np.float32)
        uv_1=uv_1.T
        suv_1=uv_1
        xyz_c=self.inverse_newcam_mtx.dot(suv_1)
        #print('camera frame coordinates:', xyz_c)
        inverse_R_mtx = np.linalg.inv(R_mtx)
        inverse_R_cam = np.linalg.inv(R_mtx_cam)
        #print('printing r cam inv mtx')
        #print(inverse_R_cam)
        #print(inverse_R_mtx)
        #xyz_c=xyz_c-self.tvec1
        XYZ_Gbl= inverse_R_cam.dot(xyz_c)
        XYZ_NE= inverse_R_mtx.dot(XYZ_Gbl)
        xyz_mod=math.sqrt((XYZ_NE.T).dot(XYZ_NE))
        XYZ_unit = XYZ_NE/xyz_mod
        k= np.array([[0,0,1]])

        cos_omega= k.dot(XYZ_unit)
        #depth_vision = alt/cos_omega[0][0]
        XYZ= 1.0 * XYZ_unit
        if return_cam:
            return XYZ, xyz_c 
        return XYZ


    def truncate(self, n, decimals=0):
        n=float(n)
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier
