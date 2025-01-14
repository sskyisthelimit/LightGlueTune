# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# Adapted by Remi Pautrat, Philipp Lindenberger

import torch
from kornia.color import rgb_to_grayscale
from torch import nn

from .utils import Extractor


def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0
    print(f"Performing simple_nms with nms_radius={nms_radius}. Initial scores shape: {scores.shape}")
    
    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    nms_result = torch.where(max_mask, scores, zeros)
    print(f"NMS completed. Output shape: {nms_result.shape}")
    return nms_result


def top_k_keypoints(keypoints, scores, k):
    print(f"Selecting top {k} keypoints. Initial keypoints shape: {keypoints.shape}, scores shape: {scores.shape}")
    if k >= len(keypoints):
        print("Number of keypoints is less than or equal to k. Returning all.")
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    print(f"Top {k} keypoints selected. Shape: {keypoints[indices].shape}")
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    print(f"Sampling descriptors. Keypoints shape: {keypoints.shape}, descriptors shape: {descriptors.shape}")
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    print(f"Descriptors sampled. Output shape: {descriptors.shape}")
    return descriptors


class SuperPoint(Extractor):
    default_conf = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
    }

    preprocess_conf = {
        "resize": 1024,
    }

    required_data_keys = ["image"]

    def __init__(self, **conf):
        super().__init__(**conf)
        print(f"Initializing SuperPoint with configuration: {conf}")
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layers are initialized here
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Detection head
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        # Descriptor head
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, self.conf.descriptor_dim, kernel_size=1, stride=1, padding=0)

        # Load weights
        checkpoint = torch.load(conf.get("ckpt_path"), map_location=torch.device("cpu"))
        print(f"Loading model from {conf.get('ckpt_path')}. Keys found in checkpoint: {list(checkpoint.keys())}")
        if "model_state_dict" in checkpoint:
            filtered_state_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if not k.startswith("bn")}
            self.load_state_dict(filtered_state_dict)
        else:
            self.load_state_dict(checkpoint)
        print(f"Model weights loaded.")

    def forward(self, data: dict) -> dict:
        print("Starting forward pass...")
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        image = data["image"]
        print(f"Image shape before preprocessing: {image.shape}")
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)
            print(f"Converted image to grayscale. New shape: {image.shape}")

        # Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        print(f"Feature map shape after encoder: {x.shape}")

        # Detection head
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        print(f"Scores shape before softmax: {scores.shape}")
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        print(f"Scores shape after processing: {scores.shape}")
        scores = simple_nms(scores, self.conf.nms_radius)
        print(f"Scores shape after NMS: {scores.shape}")

        # Border suppression
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1
            print(f"Scores shape after border suppression: {scores.shape}")

        # Keypoint extraction
        best_kp = torch.where(scores > self.conf.detection_threshold)
        print(f"Number of detected keypoints: {len(best_kp[0])}")
        scores = scores[best_kp]

        # Separate batches
        keypoints = [torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i] for i in range(b)]
        scores = [scores[best_kp[0] == i] for i in range(b)]

        # Top-k keypoints
        if self.conf.max_num_keypoints is not None:
            keypoints, scores = list(
                zip(
                    *[
                        top_k_keypoints(k, s, self.conf.max_num_keypoints)
                        for k, s in zip(keypoints, scores)
                    ]
                )
            )

        keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        print(f"Final keypoints shape: {keypoints[0].shape}")

        # Descriptor head
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        print(f"Descriptor map shape: {descriptors.shape}")

        # Extract descriptors
        descriptors = [
            sample_descriptors(k[None], d[None], 8)[0]
            for k, d in zip(keypoints, descriptors)
        ]

        print(f"Forward pass completed.")
        return {
            "keypoints": torch.stack(keypoints, 0),
            "keypoint_scores": torch.stack(scores, 0),
            "descriptors": torch.stack(descriptors, 0).transpose(-1, -2).contiguous(),
        }
