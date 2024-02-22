import cv2
import numpy as np

# The code in this file is sourced with minor edits from: 
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/plotting.py

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                         [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

def draw_skeleton(im, kpts, radius=5, kpt_line=True):
    """
    Plot keypoints on the image.

    Args:
        kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
        shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
        radius (int, optional): Radius of the drawn keypoints. Default is 5.
        kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                    for human pose. Default is True.

    Note: `kpt_line=True` currently only supports human pose plotting.
    """
    height, width, channels = im.shape
    nkpt, ndim = kpts.shape
    is_pose = nkpt == 17 and ndim == 3
    kpt_line &= is_pose
    for i, k in enumerate(kpts):
        color_k = [int(x) for x in kpt_color[i]]
        x_coord, y_coord = k[0], k[1]
        if x_coord % width != 0 and y_coord % height != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

    if kpt_line:
        ndim = kpts.shape[-1]
        for i, sk in enumerate(skeleton):
            pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
            pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
            if ndim == 3:
                conf1 = kpts[(sk[0] - 1), 2]
                conf2 = kpts[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
            if pos1[0] % width == 0 or pos1[1] % height == 0 or pos1[0] < 0 or pos1[1] < 0:
                continue
            if pos2[0] % width == 0 or pos2[1] % height == 0 or pos2[0] < 0 or pos2[1] < 0:
                continue
            cv2.line(im, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

    return im

def draw_box(im, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(im.shape) / 2 * 0.003), 2)
    tf = max(lw - 1, 1)  # font thickness
    sf = lw / 3  #  font scale    
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    sf,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return im

def draw_mask(im, mask, color =(255,125,125), alpha=(0.5)):
    mask_np = mask.astype(np.int32)
    overlay = im.copy()
    cv2.polylines(im, [mask_np], isClosed=False, color=color, thickness=2)
    cv2.fillPoly(overlay, pts=[mask_np], color=color, lineType=cv2.LINE_AA)
    im = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0) 
    return im