import os
from random import sample
import json
import torch
import h5py
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from preprocessing.data import FrameData
from preprocessing.model import ResNet101NoFC


def get_id2name_file(AGGR_JSON, SCENE_LIST):
    print("getting id2name...")
    id2name = {}
    item_ids = []
    all_scenes = SCENE_LIST
    print("Number of scenes: ", len(all_scenes))
    for scene_id in tqdm(all_scenes):
        id2name[scene_id] = {}
        aggr_file = json.load(open(AGGR_JSON.format(scene_id, scene_id)))
        for item in aggr_file["segGroups"]:
            item_ids.append(int(item["id"]))
            id2name[scene_id][int(item["id"])] = item["label"]

    return id2name


def get_label_info(SCANNET_V2_TSV):
    label2class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
                   'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
                   'refrigerator': 12, 'shower curtain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16, 'others': 17}

    # mapping
    scannet_labels = label2class.keys()
    scannet2label = {label: i for i, label in enumerate(scannet_labels)}

    lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
    lines = lines[1:]
    raw2label = {}
    for i in range(len(lines)):
        label_classes_set = set(scannet_labels)
        elements = lines[i].split('\t')
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2label[raw_name] = scannet2label['others']
        else:
            raw2label[raw_name] = scannet2label[nyu40_name]

    return raw2label, label2class


def validate_bbox(xyxy, width, height):
    x_min = xyxy[0]
    y_min = xyxy[1]
    x_max = xyxy[2]
    y_max = xyxy[3]
    fix = 5
    if x_max - x_min < fix:
        if x_min > fix:
            x_min -= fix
        elif x_max < width - fix:
            x_max += fix

    if y_max - y_min < fix:
        if y_min > fix:
            y_min -= fix
        elif y_max < height - fix:
            y_max += fix
    
    try:
        assert y_max <= height
        assert x_max <= width

    except AssertionError:
        print("ymax: ", y_max)
        print("xmax: ", x_max)
        print("height and width: {}, {}".format(height, width))


    return [x_min, y_min, x_max, y_max]


def sanitize_id_coco(
        IMAGE_ID
):
    assert len(IMAGE_ID) >= 9 and len(IMAGE_ID) <= 11

    scene_id = 'scene' + IMAGE_ID[:7]
    ann_id = IMAGE_ID[-1]
    if len(IMAGE_ID) == 11:
        object_id = IMAGE_ID[7:10]

    if len(IMAGE_ID) == 10:
        object_id = IMAGE_ID[7:9]

    if len(IMAGE_ID) == 9:
        object_id = IMAGE_ID[7:8]

    sanitized = '{}-{}_{}'.format(scene_id, object_id, ann_id)
    return sanitized, scene_id, object_id, ann_id


def get_iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the coordinates of intersection of box1 and box2. 
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    # Calculate intersection area.
    inter_area = (y2_inter - y1_inter) * (x2_inter - x1_inter)

    # Calculate the Union area.
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU    
    iou = inter_area / union_area

    return iou


def sort_by_iou(
        SAMPLE_ID,
        SAMPLE_ID_DETECTIONS,
        GT_DB
):
    """
        Takes detections for each sample id, sorts them by IOU descending, and returns a
        SAMPLE_ID_DETECTIONS dictionary with added IOU scores.
    """
    
    try:
        sorted_detections = []
        target_object_id = int(SAMPLE_ID.split('-')[1].split('_')[0])
        gt_boxes = np.array(GT_DB['box'][SAMPLE_ID])
        gt_oids = np.array(GT_DB['objectids'][SAMPLE_ID])
        target_box_idx = np.where(gt_oids == target_object_id)[0]
        target_box = gt_boxes[target_box_idx].tolist()[0]

        # CONVERT BOX FORMAT FROM XYWH TO XYXY
        for item in SAMPLE_ID_DETECTIONS:
            detected_box = item['bbox']
            item['iou'] = get_iou(detected_box, target_box)
            sorted_detections.append(item)

        sorted_detections = sorted(sorted_detections, key=lambda x: x['iou'], reverse=True)  # descending

    except:
        print("Ignored sample {}".format(SAMPLE_ID))
        sorted_detections = None

    return sorted_detections


def export_bbox_pickle_coco(
        MRCNN_DETECTIONS_PATH,
        DB_PATH,
        GT_DB_PATH,
        RESIZE=(320, 240)
):

    pickle_dir = os.path.dirname(DB_PATH)
    os.makedirs(pickle_dir, exist_ok=True)
    db = h5py.File(DB_PATH, 'w')
    assert os.path.exists(GT_DB_PATH)
    gt_db = h5py.File(GT_DB_PATH, 'r')
    assert os.path.exists(MRCNN_DETECTIONS_PATH)
    detections = json.load(open(MRCNN_DETECTIONS_PATH))
    print("validating the mask r-cnn predictions.")
    aggregations = {}  # render_id -> list of predictions
    for pred in tqdm(detections):
        x_min, y_min, w, h = pred['bbox']
        validated_bbox = [x_min, y_min, x_min + w, y_min + h]
        validated_score = round(pred['score'], 2)
        pred['bbox'] = validated_bbox
        pred['score'] = validated_score
        sample_id, _, _, _ = sanitize_id_coco(pred['image_id'])
        if sample_id in aggregations.keys():  # filter ignored_renders
            aggregations[sample_id].append(pred)
        else:
            aggregations[sample_id] = [pred]

    # Sort based on the IoU score with the gt box in that frame/render #####
    aggregations_sorted = {}
    ditch_control = 0
    print("sorting the bounding boxes based on IoU.")
    for sample_id, detections in aggregations.items():
        res = sort_by_iou(sample_id, detections, gt_db)
        if res is not None:
            aggregations_sorted[sample_id] = res
        else:
            ditch_control += 1

    assert ditch_control <= 250

    for sample_id, detections in tqdm(aggregations_sorted.items()):
        detections = list(filter(lambda x: x['iou'] >= 0.5, detections))

        boxes = []
        ious = []
        scores = []
        categories = []
        object_ids = []
        for object_id, d in enumerate(detections):
            # RESIZE [0] -> Width
            # RESIZE [1] -> Height
            # d['segmentation']['size'][1] -> Width
            # d['segmentation']['size'][0] -> Height
            scale_x = RESIZE[0] / d['segmentation']['size'][1]
            scale_y = RESIZE[1] / d['segmentation']['size'][0]
            scaled_box = [math.floor(scale_x * d['bbox'][0]), math.floor(scale_y * d['bbox'][1]), math.ceil(scale_x * d['bbox'][2]) - 1, math.ceil(scale_y * d['bbox'][3]) - 1]
            scaled_box = np.array(validate_bbox(scaled_box, width=RESIZE[0], height=RESIZE[1]))
            iou = np.array(d['iou'])
            score = np.array(d['score'])
            category = np.array(d['category_id'])
            object_id = np.array(object_id)


            boxes.append(scaled_box)
            ious.append(iou)
            scores.append(score)
            categories.append(category)
            object_ids.append(object_id)

        if len(boxes) >= 1:
            boxes = np.vstack(boxes)
            ious = np.vstack(ious)
            scores = np.vstack(scores)
            object_ids = np.vstack(object_ids)
            categories = np.vstack(categories)
            db.create_dataset('box/{}'.format(sample_id), data=boxes)
            db.create_dataset('ious/{}'.format(sample_id), data=ious)
            db.create_dataset('scores/{}'.format(sample_id), data=scores)
            db.create_dataset('objectids/{}'.format(sample_id), data=object_ids)
            db.create_dataset('categories/{}'.format(sample_id), data=categories)
    
    db.close()



def load_proper_label_img(instance_mask_path, scene_id, sample_id):    
    
    label_path = os.path.join(instance_mask_path.format(scene_id, sample_id))
    label_img = np.array(Image.open(label_path))    

    return label_img

def export_bbox_pickle_raw(
        AGGR_JSON_PATH,
        SCANNET_V2_TSV,
        INSTANCE_MASK_PATH,
        SAMPLE_LIST,
        SCENE_LIST,
        DB_PATH,
        RESIZE=(320, 240)
):
    id2name = get_id2name_file(AGGR_JSON=AGGR_JSON_PATH, SCENE_LIST=SCENE_LIST)
    raw2label, label2class = get_label_info(SCANNET_V2_TSV=SCANNET_V2_TSV)
    pickle_dir = os.path.dirname(DB_PATH)
    os.makedirs(pickle_dir, exist_ok=True)
    
    print("db location: {}".format(DB_PATH))
    print("exporting image bounding boxes...")
        
    db = h5py.File(DB_PATH, 'w')
    for gg in tqdm(SAMPLE_LIST):
        sample_id = gg['sample_id']
        scene_id = gg['scene_id']
        object_id = gg['object_id']
        ann_id = gg['ann_id']

        try:
            label_img = load_proper_label_img(INSTANCE_MASK_PATH, scene_id, sample_id)
            scale_x = RESIZE[0] / label_img.shape[1]
            scale_y = RESIZE[1] / label_img.shape[0]
        except FileNotFoundError as fnfe:
            # print(fnfe)
            continue

        labels = np.unique(label_img)
        bbox = []
        object_ids = []
        sem_labels = []

        for label in labels:
            if label == 0: continue
            raw_name = id2name[scene_id][label - 1]
            sem_label = raw2label[raw_name]
            if raw_name in ["floor", "wall", "ceiling"]: continue
            target_coords = np.where(label_img == label)
            x_max, y_max = np.max(target_coords[1], axis=0), np.max(target_coords[0], axis=0)
            x_min, y_min = np.min(target_coords[1], axis=0), np.min(target_coords[0], axis=0)
            bbox_scaled = [math.floor(x_min * scale_x), math.floor(y_min * scale_y), math.ceil(x_max * scale_x) - 1, math.ceil(y_max * scale_y) - 1]
            bbox_validated = validate_bbox(bbox_scaled, RESIZE[0], RESIZE[1])
            bbox.append(np.array(bbox_validated, dtype=np.float))
            object_ids.append(np.array(label - 1, dtype=np.uint8))
            sem_labels.append(np.array(sem_label, dtype=np.uint8))

        if len(bbox) >= 1:
            bbox = np.vstack(bbox)
            oids = np.vstack(object_ids)
            slabels = np.vstack(sem_labels)
            write_key = '{}-{}_{}'.format(scene_id, object_id, ann_id)
            db.create_dataset('box/{}'.format(write_key), data=bbox)
            db.create_dataset('objectids/{}'.format(write_key), data=oids)
            db.create_dataset('semlabels/{}'.format(write_key), data=slabels)

    db.close()

    print("Created boxes.")


def export_image_features(
        KEY_FORMAT,
        IMAGE,
        DB_PATH,
        BOX,
        SAMPLE_LIST,
        IGNORED_SAMPLES,
        DEVICE,
        RESIZE
):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    fd_train = FrameData(
        ignored_samples=IGNORED_SAMPLES,
        key_format=KEY_FORMAT,
        resize=RESIZE,
        frame_path=IMAGE,
        db_path=DB_PATH,
        box=BOX,
        input_list=SAMPLE_LIST,
        transforms=normalize
    )

    conf = {
        'batch_size': 16,
        'num_workers': 6
    }

    data_loader = DataLoader(fd_train, collate_fn=fd_train.collate_fn, **conf)
    model = ResNet101NoFC(pretrained=True, progress=True, device=DEVICE, mode='frame2feat').to(DEVICE)
    model.eval()

    print("Frame feature extraction started.")

    db = h5py.File(DB_PATH, 'a')
    target_dir = os.path.dirname(DB_PATH)
    assert os.path.exists(target_dir)

    for i, f in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            tensor_list, bbox_list, bbox_ids, sample_id_list = f
            frame_features = model(tensor_list, None, None).to('cuda')
            batch_size = len(tensor_list)
            for i in range(batch_size):
                k = '{}'.format(sample_id_list[i])
                db.create_dataset('globalfeat/{}'.format(k), data=frame_features[i].detach().cpu().numpy())

    print("Saved extracted features.")
    db.close()

    return None


def export_bbox_features(
        IGNORED_SAMPLES,
        KEY_FORMAT,
        IMAGE,
        DB_PATH,
        BOX,
        SAMPLE_LIST,
        DEVICE,
        RESIZE
):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    fd_train = FrameData(
        ignored_samples=IGNORED_SAMPLES,
        key_format=KEY_FORMAT,
        resize=RESIZE,
        frame_path=IMAGE,
        db_path=DB_PATH,
        box=BOX,
        input_list=SAMPLE_LIST,
        transforms=normalize
    )

    conf = {
        'batch_size': 64,
        'num_workers': 6
    }

    data_loader = DataLoader(fd_train, collate_fn=fd_train.collate_fn, **conf)
    model = ResNet101NoFC(pretrained=True, progress=True, device=DEVICE, mode='bbox2feat').to(DEVICE)
    model.eval()

    db = h5py.File(DB_PATH, 'a')
    target_dir = os.path.dirname(DB_PATH)
    assert os.path.exists(target_dir)

    print("Box feature extraction started.")

    for i, f in enumerate(tqdm(data_loader)):
        tensor_list, bbox_list, bbox_ids, sample_id_list = f

        with torch.no_grad():
            batch_feats = model(tensor_list, bbox_list, bbox_ids)
            batch_size = len(tensor_list)
            for i in range(batch_size):
                frame_object_features = batch_feats[i]
                sample_id = sample_id_list[i]
                object_ids = np.array(list(frame_object_features.keys()), dtype=np.uint8)
                features = np.vstack([item.squeeze().cpu().numpy() for item in list(frame_object_features.values())])
                db.create_dataset('boxobjectid/{}'.format(sample_id), data=object_ids)
                db.create_dataset('boxfeat/{}'.format(sample_id), data=features)        

    print("Saved extracted features.")
    db.close()
    
    return None
