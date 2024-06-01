import MatterSim
import math
import numpy as np
from PIL import Image
import json
import os
from collections import defaultdict

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
    for index, rgb in enumerate(rgbs):
        image_filename = '{}_viewpoint_{}_res_{}x{}.jpg'.format(viewpoint, index, image_resolution[0], image_resolution[1])
        image_path = os.path.join(scan_root, scan, 'matterport_skybox_images', image_filename)
        save_rgb(image_path, rgb)
    

sim = build_sim()

anno_root = '../../datasets_docker/R2R/annotations/'
scan_dir = '/root/mount/Matterport3DSimulator/data/v1/scans/'

processed_viewpoint = defaultdict(list)

# for filename in ['R2R_train_enc.json', 'R2R_val_seen_enc.json', 'R2R_val_unseen_enc.json']:
for filename in ['R2R_val_seen_enc.json', 'R2R_val_unseen_enc.json']:
    filepath = os.path.join(anno_root, filename)
    print("Processing file: {} ...".format(filepath))
    json_objects = load_enc_json(filepath)

    for path_item in json_objects:
        scan_id = path_item['scan']
        for viewpoint_id in path_item['path']:
            if viewpoint_id in processed_viewpoint[scan_id]:
                print('Skipped Scan {} Viewpoint {}'.format(scan_id, viewpoint_id))
                continue
            rgbs = get_sim_viewpoint_rgbs(sim, scan_id, viewpoint_id)

            save_rgb_list(scan_dir, rgbs, scan_id, viewpoint_id)

            processed_viewpoint[scan_id].append(viewpoint_id)

            print("Saved to Scan {} Viewpoint {}".format(scan_id, viewpoint_id))

    print('\007')
    import pdb; pdb.set_trace()
