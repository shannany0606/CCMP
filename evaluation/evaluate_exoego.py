
import json
import argparse
from pycocotools import mask as mask_utils
import numpy as np
import tqdm
from sklearn.metrics import balanced_accuracy_score

import utils

CONF_THRESH = 0.5
H, W = 480, 480 # resolution for evalution

# Eight high-level domains and their identifying keywords (lowercase)
DOMAIN_KEYWORDS = {
    "cooking": [
        "cook", "egg", "salad", "noodles", "pasta", "coffee", "milk", "tomato", "tea", "dinner", "plate",
        # Kitchen utensils
        "bowl", "spoon", "knife", "fork", "spatula", "whisk", "ladle", "chopstick", "tong",
        "pot", "pan", "kettle", "skillet", "wok", "cup", "mug", 
        "chopping board", "chopping_board", "cutting board", "grater", "peeler", "strainer", "colander", "sieve",
        "measuring spoon", "measuring cup", "scale",
        "mortar", "pestle", "scissors", "scissor", "dipper", "scoop", "skimmer",
        "rolling pin",
        # Ingredients and seasonings
        "onion", "garlic", "ginger", "celery", "cucumber", "carrot", "pepper", "chili", "basil",
        "salt", "sugar", "honey", "butter", "cheese", "curd", "cream",
        "oil", "vinegar", "sauce", "ketchup", "mustard", "syrup", "spice", "oregano", "cinnamon",
        "flour", "nut", "almond", "peanut", "sesame", "groundnut",
        "beef", "meat", "cherry", "lemon", "olive", "vegetable", "parsley", "cilantro", "scallion", "ciliary",
        "coriander", "molasses", "beer",
        "fish", "sriracha", "paprika", "turmeric", "curry",
        "omelet", "omelette", "fry", "fried", "scrambled", "boil", "stir",
        # Kitchen supplies and equipment
        "napkin", "towel", "tissue", "paper towel", "cloth", "rag",
        "lighter", "matches", "dispenser", "holder", "organizer", "bin", "waste", "trash", "thrash",
        "table", "stool", "tray", "storage", "container", "bottle", "pack", "carton",
        "dining", "kitchen", "recipe", "instruction",
        "chafing dish", "espresso machine", "timer", "knob", "control",
        "bucket", "trolley", "jug", "jar", "can", "drain", "wrap",
        "pakkad", "soap", "stainer", "steel thong", "matchbox", "beaker", "blue basket", "stainless steel bowel",
        "cell phone", "clipboard", "wooden clip board"
    ],
    "health": [
        "covid", "test", "swab", "cassette", "antigen", "rapid test", "nasal",
        "extraction buffer", "solution tube", "sterile", "cpr", "manikin", "dummy",
        "medical", "health", "clinic", "patient", "toothbrush", "quality card"
    ],
    "bike repair": [
        "wheel", "chain", "bike", "bicycle", "tire", "tyre", "tube", "pump",
        "brake", "pedal", "lever", "wrench", "spanner", "clamp",
        "handlebar", "fork", "seat", "stay", "cable", "sprocket", "caliper",
        "valve", "rim", "spoke"
    ],
    "music": [
        "violin", "piano", "guitar", "music"
    ],
    "basketball": [
        "basketball", "hoop"
    ],
    "soccer": [
        "soccer"
    ],
}

def categorize_object(obj_name: str) -> str:
    """Return one of the eight domain names based on keyword match; 'other' if none.

    Matching is case-insensitive and checks if any keyword is a substring of the
    provided object name. The mapping is heuristic but sufficient for aggregation.
    """
    name_l = str(obj_name).lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in name_l:
                return domain
    return "other"

def evaluate_take(gt, pred):
    
    IoUs = []
    ShapeAcc = []
    ExistenceAcc = []
    LocationScores = []

    ObjExist_GT = []
    ObjExist_Pred = []

    ObjSizeGT = []
    ObjSizePred = []
    IMSize = []

    # Track per-object IoUs to enable domain-wise aggregation across takes
    ObjToIoUs = {}

    for object_id in gt['masks'].keys():
        ego_cams = [x for x in gt['masks'][object_id].keys() if 'aria' in x]
        if len(ego_cams) < 1:
            continue
        assert len(ego_cams) == 1
        EGOCAM = ego_cams[0]

        EXOCAMS = [x for x in gt['masks'][object_id].keys() if 'aria' not in x]
        for exo_cam in EXOCAMS:
            gt_masks_ego = {}
            gt_masks_exo = {}
            pred_masks_ego = {}

            if EGOCAM in gt["masks"][object_id].keys():
                gt_masks_ego = gt["masks"][object_id][EGOCAM]
            if exo_cam in gt["masks"][object_id].keys():
                gt_masks_exo = gt["masks"][object_id][exo_cam]
            if object_id in pred["masks"].keys() and f'{exo_cam}_{EGOCAM}' in pred["masks"][object_id].keys():
                pred_masks_ego = pred["masks"][object_id][f'{exo_cam}_{EGOCAM}']

            for frame_idx in gt_masks_exo.keys():
                
                if int(frame_idx) not in gt["annotated_frames"][object_id][EGOCAM]:
                    continue
                
                if not frame_idx in gt_masks_ego:
                    gt_mask = None
                    gt_obj_exists = 0
                else:
                    gt_mask = mask_utils.decode(gt_masks_ego[frame_idx])
                    # reshaping without padding for evaluation
                    gt_mask = utils.reshape_img_nopad(gt_mask)

                    gt_obj_exists = 1

                try:
                    pred_mask = mask_utils.decode(pred_masks_ego[frame_idx]["pred_mask"])
                except:
                    breakpoint()

                pred_obj_exists = int(pred_masks_ego[frame_idx]["confidence"] > CONF_THRESH)

                if gt_obj_exists:
                    # iou and shape accuracy
                    try:
                        iou, shape_acc = utils.eval_mask(gt_mask, pred_mask)
                    except:
                        breakpoint()

                    # compute existence acc i.e. if gt == pred == ALL ZEROS or gt == pred == SOME MASK
                    ex_acc = utils.existence_accuracy(gt_mask, pred_mask)

                    # # location accuracy
                    location_score = utils.location_score(gt_mask, pred_mask, size=(H, W))

                    IoUs.append(iou)
                    ShapeAcc.append(shape_acc)
                    ExistenceAcc.append(ex_acc)
                    LocationScores.append(location_score)

                    # collect per-object IoUs
                    if object_id not in ObjToIoUs:
                        ObjToIoUs[object_id] = []
                    ObjToIoUs[object_id].append(iou)

                    ObjSizeGT.append(np.sum(gt_mask).item())
                    ObjSizePred.append(np.sum(pred_mask).item())
                    IMSize.append(list(gt_mask.shape[:2]))

                ObjExist_GT.append(gt_obj_exists)
                ObjExist_Pred.append(pred_obj_exists)

    IoUs = np.array(IoUs)
    ShapeAcc = np.array(ShapeAcc)
    ExistenceAcc = np.array(ExistenceAcc)
    LocationScores = np.array(LocationScores)

    return IoUs.tolist(), ShapeAcc.tolist(), ExistenceAcc.tolist(), LocationScores.tolist(), \
            ObjExist_GT, ObjExist_Pred, ObjSizeGT, ObjSizePred, IMSize, ObjToIoUs

def validate_predictions(gt, preds):

    assert "exo-ego" in preds
    preds = preds["exo-ego"]

    assert type(preds) == type({})
    for key in ["results"]:
        assert key in preds.keys()

    assert len(preds["results"]) == len(gt["annotations"])
    for take_id in gt["annotations"]:
        assert take_id in preds["results"]

        for key in ["masks", "subsample_idx"]:
            assert key in preds["results"][take_id]

        # check objs
        assert len(preds["results"][take_id]["masks"]) == len(gt["annotations"][take_id]["masks"])
        for obj in gt["annotations"][take_id]["masks"]:
            assert obj in preds["results"][take_id]["masks"], f"{obj} not in pred {take_id}"

            ego_cam = None
            exo_cams = []
            for cam in gt["annotations"][take_id]["masks"][obj]:
                if 'aria' in cam:
                    ego_cam = cam
                else:
                    exo_cams.append(cam)
            try:
                assert not ego_cam is None
            except:
                continue
            try:
                assert len(exo_cams) > 0
            except:
                continue

            for cam in exo_cams:
                try:
                    assert f"{cam}_{ego_cam}" in preds["results"][take_id]["masks"][obj]
                except:
                    breakpoint()

                for idx in gt["annotations"][take_id]["masks"][obj][cam]:
                    assert idx in preds["results"][take_id]["masks"][obj][f"{cam}_{ego_cam}"]

                    for key in ["pred_mask", "confidence"]:
                        assert key in preds["results"][take_id]["masks"][obj][f"{cam}_{ego_cam}"][idx]

def evaluate(gt, preds):

    validate_predictions(gt, preds)
    preds = preds["exo-ego"]

    total_iou = []
    total_shape_acc = []
    total_existence_acc = []
    total_location_scores = []

    total_obj_sizes_gt = []
    total_obj_sizes_pred = []
    total_img_sizes = []

    # total_obj_exists_gt = []
    # total_obj_exists_pred = []

    # per-domain IoU accumulator
    domain_to_ious = {k: [] for k in list(DOMAIN_KEYWORDS.keys()) + ["other"]}

    for take_id in tqdm.tqdm(gt["annotations"]):

        ious, shape_accs, existence_accs, location_scores, obj_exist_gt, obj_exist_pred, \
            obj_size_gt, obj_size_pred, img_sizes, obj_to_ious = evaluate_take(gt["annotations"][take_id], 
                                                                              preds["results"][take_id])

        total_iou += ious
        total_shape_acc += shape_accs
        total_existence_acc += existence_accs
        total_location_scores += location_scores

        total_obj_sizes_gt += obj_size_gt
        total_obj_sizes_pred += obj_size_pred
        total_img_sizes += img_sizes

        # total_obj_exists_gt += obj_exist_gt
        # total_obj_exists_pred += obj_exist_pred
    
        # aggregate per-domain IoUs for this take
        for obj_id, obj_ious in obj_to_ious.items():
            domain = categorize_object(obj_id)
            domain_to_ious.setdefault(domain, [])
            domain_to_ious[domain] += obj_ious
    
    print('the result of exoego: ')
    print('TOTAL EXISTENCE ACC: ', np.mean(total_existence_acc))
    # print('TOTAL EXISTENCE BALANCED ACC: ', balanced_accuracy_score(total_obj_exists_gt, total_obj_exists_pred))
    print('TOTAL IOU: ', np.mean(total_iou))
    print('TOTAL LOCATION SCORE: ', np.mean(total_location_scores))
    print('TOTAL SHAPE ACC: ', np.mean(total_shape_acc))

    # Print per-domain mIoU
    print('DOMAIN-WISE mIoU:')
    for domain, ious in domain_to_ious.items():
        if len(ious) == 0:
            print(f'  {domain}: N/A (no samples)')
        else:
            print(f'  {domain}: {np.mean(ious)}')

    # Size-binned mIoU by GT mask area percentage of original image
    sizes = np.array(total_obj_sizes_gt, dtype=np.float64)
    img_pixels = np.array([h*w for h, w in total_img_sizes], dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        size_pct = (sizes / img_pixels) * 100.0

    print('SIZE-BIN mIoU (GT area percentage):')
    ious_np = np.array(total_iou, dtype=np.float64)
    bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        sel = (size_pct >= lo) & (size_pct < hi)
        vals = ious_np[sel]
        if vals.size == 0:
            print(f'  {lo}-{hi}%: N/A (no samples)')
        else:
            print(f'  {lo}-{hi}%: {np.mean(vals)}')
    sel = size_pct >= 0.5
    vals = ious_np[sel]
    if vals.size == 0:
        print('  0.5+%: N/A (no samples)')
    else:
        print(f'  0.5+%: {np.mean(vals)}')
    sel = size_pct >= 1
    vals = ious_np[sel]
    if vals.size == 0:
        print('  1+%: N/A (no samples)')
    else:
        print(f'  1+%: {np.mean(vals)}')

def main(args):

    # load gt and pred jsons
    with open(args.gt_file, 'r') as fp:
        gt = json.load(fp)

    with open(args.pred_file, 'r') as fp:
        preds = json.load(fp)

    # evaluate
    evaluate(gt, preds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-file', type=str, required=True, 
                            help="path to json with gt annotations")
    parser.add_argument('--pred-file', type=str, required=True,
                            help="")
    args = parser.parse_args()
    
    main(args)