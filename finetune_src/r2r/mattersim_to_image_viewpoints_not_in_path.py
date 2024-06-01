import MatterSim
import math
import numpy as np
from PIL import Image
import json
import os
from collections import defaultdict
import pickle


# image_resolution = (640, 480)
image_resolution = (960, 720)

def build_sim():
    global image_resolution
    sim = MatterSim.Simulator()
    sim.setCameraResolution(image_resolution[0], image_resolution[1])
    sim.setPreloadingEnabled(True)
    sim.setDepthEnabled(False)
    sim.setBatchSize(1)
    sim.setCacheSize(200)

    VFOV = 60
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)

    scan_data_dir = '/root/mount/Matterport3DSimulator/data/v1/scans'
    sim.setDatasetPath(scan_data_dir)
    connectivity_dir = '/root/mount/Matterport3DSimulator/connectivity'
    sim.setNavGraphPath(connectivity_dir)

    sim.initialize()
    return sim

def get_sim_rgb(sim, scan, viewpoint):
    sim.newEpisode([scan], [viewpoint], [0], [0])
    rgb = np.array(sim.getState()[0].rgb)
    return rgb

def save_rgb(image_path, rgb):
    im = Image.fromarray(rgb)
    im.save(image_path)

def get_sim_viewpoint_rgbs(sim, scan, viewpoint):
    rgb_list = list()
    for ix in range(36):
        if ix == 0:
            sim.newEpisode([scan], [viewpoint], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])
        rgb = np.array(sim.getState()[0].rgb)
        # todo: bgr -> rgb
        rgb = rgb[:, :, ::-1]
        rgb_list.append(rgb)
    return rgb_list

def load_enc_json(filepath):
    with open(filepath, 'r') as f:
        new_data = json.load(f)
    return new_data

def save_rgb_list(scan_root, rgbs, scan, viewpoint):
    global image_resolution
    this_scan_dir = os.path.join(scan_root, scan)
    skybox_dir = os.path.join(this_scan_dir, 'matterport_skybox_images')
    if not os.path.exists(this_scan_dir):
        os.makedirs(this_scan_dir)
    if not os.path.exists(skybox_dir):
        os.makedirs(skybox_dir)
    for index, rgb in enumerate(rgbs):
        image_filename = '{}_viewpoint_{}_res_{}x{}.jpg'.format(viewpoint, index, image_resolution[0], image_resolution[1])
        image_path = os.path.join(scan_root, scan, 'matterport_skybox_images', image_filename)
        # assert not os.path.exists(image_path)
        save_rgb(image_path, rgb)
    

sim = build_sim()

anno_root = '../../datasets_docker/R2R/annotations/'
scan_dir = '/root/mount/Matterport3DSimulator/data/v1/scans/'


scan_dir_for_missing_vps = '/root/mount/Matterport3DSimulator/data/v1/scans_for_missing_vps/'

processed_viewpoint = defaultdict(list)

with open('viewpoints_not_in_path.pkl', 'rb') as f:
    viewpoints_not_in_path = pickle.load(f)


for scan in viewpoints_not_in_path:
    for viewpoint in viewpoints_not_in_path[scan]:
        if viewpoint in processed_viewpoint[scan]:
            print('Skipped Scan {} Viewpoint {}'.format(scan, viewpoint))
            continue
        try:
            rgbs = get_sim_viewpoint_rgbs(sim, scan, viewpoint)
        except ValueError as e:
            print(e)
            print('Skipped Scan {} Viewpoint {} For Error'.format(scan, viewpoint))
        save_rgb_list(scan_dir_for_missing_vps, rgbs, scan, viewpoint)
        processed_viewpoint[scan].append(viewpoint)

        print("Saved to Scan {} Viewpoint {}".format(scan, viewpoint))
