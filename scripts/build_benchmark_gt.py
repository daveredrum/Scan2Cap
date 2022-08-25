import os
import json
import argparse

import numpy as np

from tqdm import tqdm

DATA_ROOT = "/cluster/balrog/dchen/ScanRefer/data/" # TODO change this
SCANNET_ROOT = os.path.join(DATA_ROOT, "scannet")

def get_scannet_scene_list(args):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(SCANNET_ROOT, "meta_data", "scannetv2_{}.txt".format(args.split)))])
    scene_list = [s for s in scene_list if s.split("_")[-1] == "00"]

    return scene_list

def get_3d_box(box_size, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''

    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    corners_3d = np.vstack([x_corners,y_corners,z_corners])
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d

def load_scanrefer(args):
    print("loading ScanRefer {}...".format(args.split))
    with open(os.path.join(DATA_ROOT, "ScanRefer_filtered_{}.json".format(args.split))) as f:
        data = json.load(f)

    return data

def load_scene_data(args):
    print("loading scene data...")
    scene_list = get_scannet_scene_list(args)

    scene_data = {}
    for scene_id in tqdm(scene_list):
        scene_data[scene_id] = {}
        data = np.load(os.path.join(SCANNET_ROOT, "scannet_data", scene_id)+"_aligned_bbox.npy")
        for i in range(data.shape[0]):
            box_6_deg, box_id = data[i, :-1], data[i, -1]

            box_corners = get_3d_box(box_6_deg[3:6], box_6_deg[:3])
            scene_data[scene_id][str(int(box_id))] = box_corners

    return scene_data

def load_data(args):
    scanrefer = load_scanrefer(args)
    scene_data = load_scene_data(args)

    return scanrefer, scene_data

def build_gt(args):
    scanrefer, scene_data = load_data(args)

    print("building GTs...")
    new = []
    for data in tqdm(scanrefer):
        scene_id = data["scene_id"]
        object_id = data["object_id"]

        bbox = scene_data[scene_id][object_id]
        data["bbox"] = bbox.tolist()

        new.append(data)

    with open(os.path.join(DATA_ROOT, "ScanRefer_filtered_{}_gt_bbox.json".format(args.split)), "w") as f:
        json.dump(new, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["train", "val", "test"])
    args = parser.parse_args()

    build_gt(args)
