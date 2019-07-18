"""
Questions
* What are these variables for?
  * voc_color
  * _eig_val
  * _eig_vec
"""

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

# change the class name, class attributes, and tthe lines with "#@" on them
class Roadrunner(data.Dataset):
    num_classes = 3
    default_resolution = [210, 160]
    mean = np.array([0.41204423, 0.30953058, 0.27727498], dtype=np.float32).reshape(
        1, 1, 3
    )
    std = np.array([0.43140923, 0.30728389, 0.29881408], dtype=np.float32).reshape(
        1, 1, 3
    )

    def __init__(self, opt, split):
        super().__init__()
        name = self.__class__.__name__.lower()
        self.data_dir = os.path.join(opt.data_dir, name)
        self.img_dir = os.path.join(self.data_dir, split)
        self.annot_path = os.path.join(self.data_dir, f"{split}.json")
        self.max_objs = 10  # @
        self.class_name = ["roadrunner", "coyote", "car"]  # @
        self._valid_ids = list(range(1, len(self.class_name) + 1))
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [
            (v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32)
            for v in range(1, self.num_classes + 1)
        ]
        self._data_rng = np.random.RandomState(521)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array(
            [
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938],
            ],
            dtype=np.float32,
        )

        self.split = split
        self.opt = opt

        print(f"==> initializing {name} {split} data.")
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print(f"Loaded {split} {self.num_samples} samples")

    def _to_float(self, x):
        return float(f"{x:.2f}")

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float(f"{score:.2f}"),
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(
            self.convert_eval_format(results),
            open(f"{save_dir}/results.json", "w"),
        )

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes(f"{save_dir}/results.json")
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
