
import fnmatch
import json
import os

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def encode_gt(mask_dir):
    """Given a path to a directory of ground-truth image segmentation masks,
    encodes them into the COCO annotations format using the COCO API.
    GT segmasks are 3D Numpy arrays of shape (n, h, w) for n predicted instances,
    in case of overlapping masks.
    These MUST be the same size as predicted masks, ensure this by using masks returned from
    model.load_image_gt.
    mask_dir: str, directory in which GT masks are stored. Avoid relative
        paths if possible.
    """
    # Constructing GT annotation file per COCO format:
    # http://cocodataset.org/#download
    gt_annos = {
        "images": [],
        "annotations": [],
        "categories": [{"name": "person", "id": 1, "supercategory": "person"}],
    }

    
    
    tmp = fnmatch.filter(os.listdir(mask_dir), "*.npy")
    N = sorted(tmp, key=lambda x: int(os.path.splitext(x)[0]))
    
    for df in N:
        i = int(df[:-4])
        # load image
        mask_name = "{}.npy".format(i) # use "{:012d}.npy" for COCO style file name
        im_name = "{}.png".format(i) # use {:012d}.jpg for COCO
        data = np.load(os.path.join(mask_dir, mask_name),allow_pickle=True).item()
        I = data["mask"]
        im_anno = {
            "id": i,
            "width": int(I.shape[1]),
            "height": int(I.shape[0]),
            "file_name": im_name,
        }

        gt_annos["images"].append(im_anno)

        # leaving license, flickr_url, coco_url, date_captured
        # fields incomplete

        # mask each individual object
        # NOTE: We assume these masks do not include backgrounds.
        # This means that the 1st object instance will have index 0!
        for val in range(I.shape[-1]):
            # get binary mask
            bin_mask = I[:, :, val].astype(bool).astype(np.uint8)
            instance_id = i * 100 + (
                val + 1
            )  # create id for instance, increment val
            # find bounding box
            def bbox2(img):
                rows = np.any(img, axis=1)
                cols = np.any(img, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                return int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)

            # encode mask
            encode_mask = mask.encode(np.asfortranarray(bin_mask))
            encode_mask["counts"] = encode_mask["counts"].decode("ascii")
            size = int(mask.area(encode_mask))
            x, y, w, h = bbox2(bin_mask)

            instance_anno = {
                "id": instance_id,
                "image_id": i,
                "category_id": 1, #use data['class'] to map
                "segmentation": encode_mask,
                "area": size,
                "bbox": [x, y, w, h],
                "iscrowd": 0,
            }

            gt_annos["annotations"].append(instance_anno)

    anno_path = os.path.join(mask_dir, "annos_gt.json")
    json.dump(gt_annos, open(anno_path, "w+"))
    print("successfully wrote GT annotations to", anno_path)

import sys
args = sys.argv[1:]
if len(args)==0 or len(args)>1:
    print("provide only the mask folder")
    sys.exit()

encode_gt(args[0])
