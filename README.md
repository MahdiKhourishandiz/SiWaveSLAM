<p align="center">

  <h1 align="center"> Siamese Wavelet SLAM: Real-Time 3D Mapping for Autonomous Vehicles with Signed Distance Function and Neural Point</h1>


![alt text](https://github.com/MahdiKhourishandiz/SiWaveSLAM/blob/main/paper%20overview/overview.png)





## Abstract

We introduce SiWaveSLAM, an autonomous vehicle LiDAR-based SLAM system that utilizes implicit neural representations and SDFs to construct high-fidelity, scalable 3D maps in dynamic scenes. Voxel hashing is used for indexing of neural points to support fast querying and map scalability. Loop-closure detection and accurate environmental feature detection are enabled by a siamese neural network with global consistency improvements. Besides, we propose a signed distance function-based point registration approach to improve map alignment accuracy and inhibit error drift. Wavelet transforms-based multi-scale feature extraction enhances the robustness of localization and alignment in challenging environments. Quantitative experiments with large scales on benchmarking datasets verify the effectiveness of our method with an Absolute Trajectory Error of 0.90 m for looped sequences of the KITTI dataset and 4.50 m for the MulRan dataset. In addition, our approach has an average relative translational drift of 0.51% on KITTI, superior to state-of-the-art SLAM methods in odometric accuracy. 



## Installation


### 1. Clone the repository

```
git clone https://github.com/MahdiKhourishandiz/SiWaveSLAM.git
cd SiWaveSLAM
```


### 2. download KITTI Dataset
```
Go to https://www.cvlibs.net/datasets/kitti/user_login.php
and Download odometry data set (velodyne laser data, 80 GB)
```

### 3. Change your KITTI Dataset
```
Change your KITTI Dataset path in the config file.
config\lidar_slam\kitti_dataset.yaml
```


### 4. Install requirements
```
pip install -r requirements.txt
```

## 5. Run SiWaveSLAM
```
python SiWaveSLAM.py
```
