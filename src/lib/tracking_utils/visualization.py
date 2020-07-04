import numpy as np
import cv2


def tlwhs_to_tlbrs(tlwhs):
    tlbrs = np.copy(tlwhs)
    if len(tlbrs) == 0:
        return tlbrs
    tlbrs[:, 2] += tlwhs[:, 0]
    tlbrs[:, 3] += tlwhs[:, 1]
    return tlbrs


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def resize_image(image, max_size=800):
    if max(image.shape[:2]) > max_size:
        scale = float(max_size) / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def plot_boxes(image, frame_id=0, fps=0., boxes=None, type='tlbr', color=(255,255,255), line_thickness=0):

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    if line_thickness == 0: # if it is not defined
        line_thickness = max(1, int(image.shape[1] / 500.))*2

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f' % (frame_id, fps),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, color, thickness=text_thickness)

    # draw boxes
    for i, box in enumerate(boxes):
        if type == 'tlbr':
            x1, y1, x2, y2 = box
            intbox = tuple(map(int, (x1, y1, x2, y2)))
        elif type == 'tlwh':
            x1, y1, w, h = box
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)

    return im


def plot_FP(image, tlwhs, obj_ids, acc_frame, frame_id=0, fps=0., color=(0,0,255)):

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    # draw match, FP, SWITCH
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = 'FP_{}'.format(int(obj_id))

        # draw FP
        mot_type = acc_frame[acc_frame.HId.eq(obj_id)].Type.values[0]
        if mot_type == 'FP':
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=-1)
            cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 0),
                        thickness=4)  # green

    return im


def plot_FN(image, tlwhs, obj_ids, acc_frame, evaluator, frame_id=0, fps=0., color=(0,0,255)):

    im = np.ascontiguousarray(np.copy(image))
    # im_h, im_w = im.shape[:2]

    # draw FN=MISS (i.e. boxes that are missed)

    gt_objs = evaluator.gt_frame_dict.get(frame_id+1, [])
    from tracking_utils.io import unzip_objs
    gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

    miss_rows = acc_frame[acc_frame.Type.eq('MISS')]
    miss_OIds = miss_rows.OId.values
    for miss_OId in miss_OIds:

        x1, y1, w, h = gt_tlwhs[gt_ids==miss_OId][0] # 2d array -> 1d array
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        color = (0, 255,255) # yellow overlay

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=-1)

    return im

def plot_all_GT(image, evaluator, scores=None, frame_id=0, fps=0., color=(0,0,255)):

    im = np.ascontiguousarray(np.copy(image))

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1

    gt_objs = evaluator.gt_frame_dict.get(frame_id+1, [])
    from tracking_utils.io import unzip_objs
    gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

    for (gt_id, gt_tlwh) in zip(gt_objs, gt_tlwhs):
        x1, y1, w, h = gt_tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        # if intbox[0] < 0 or intbox[1] < 0 or intbox[2] > 1920 or intbox[3] > 1080:
        #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     print(intbox)
        #     print(frame_id)
        #     print(gt_id)

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=3)
        cv2.putText(im, '{}'.format(gt_id[1]), (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 0),
                    thickness=text_thickness)

    return im


def plot_hypotheses(image, tlwhs, obj_ids, color=(255,0,0)):

    im = np.ascontiguousarray(np.copy(image))
    line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = max(1, image.shape[1] / 1600.)

    # draw all hypotheses
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = 'HId_{}'.format(int(obj_id))

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, color,
                    thickness=3)  # blue

    return im


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im


def plot_trajectory(image, tlwhs, track_ids):
    image = image.copy()
    for one_tlwhs, track_id in zip(tlwhs, track_ids):
        color = get_color(int(track_id))
        for tlwh in one_tlwhs:
            x1, y1, w, h = tuple(map(int, tlwh))
            cv2.circle(image, (int(x1 + 0.5 * w), int(y1 + h)), 2, color, thickness=2)

    return image


def plot_detections(image, tlbrs, scores=None, color=(255, 0, 0), ids=None):
    im = np.copy(image)
    text_scale = max(1, image.shape[1] / 800.)
    thickness = 2 if text_scale > 1.3 else 1
    for i, det in enumerate(tlbrs):
        x1, y1, x2, y2 = np.asarray(det[:4], dtype=np.int)
        if len(det) >= 7:
            label = 'det' if det[5] > 0 else 'trk'
            if ids is not None:
                text = '{}# {:.2f}: {:d}'.format(label, det[6], ids[i])
                cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                            thickness=thickness)
            else:
                text = '{}# {:.2f}'.format(label, det[6])

        if scores is not None:
            text = '{:.2f}'.format(scores[i])
            cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                        thickness=thickness)

        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

    return im
