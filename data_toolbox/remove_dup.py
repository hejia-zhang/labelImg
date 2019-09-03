import os
from PIL import Image

WIDTH = 1280
HEIGHT = 720


def yolov3_to_minmax(yolov3):
    x = float(yolov3[0])
    y = float(yolov3[1])
    w = float(yolov3[2])
    h = float(yolov3[3])
    minx = (x * WIDTH - w * WIDTH / 2)
    maxx = (x * WIDTH + w * WIDTH / 2)
    miny = (y * HEIGHT - h * HEIGHT / 2)
    maxy = (y * HEIGHT + h * HEIGHT / 2)
    minmax = [minx, miny, maxx, maxy]

    return minmax


def computeIOU(box0, box1):
    # determine
    xA = max(box0[0], box1[0])
    yA = max(box0[1], box1[1])
    xB = min(box0[2], box1[2])
    yB = min(box0[3], box1[3])
    interArea = abs(max(xB - xA, 0) * max(yB - yA, 0))
    if interArea == 0:
        return 0
    box0Area = abs((box0[2] - box0[0]) * (box0[3] - box0[1]))
    box1Area = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    iou = interArea / float(box0Area + box1Area - interArea)
    return iou


def run(label_dir, label_out_dir):
    os.makedirs(label_out_dir, exist_ok=True)

    labels = os.listdir(label_dir)
    for label_file in labels:
        if 'classes' in label_file:
            continue
        new_labels = []
        with open(os.path.join(label_dir, label_file), 'r') as f:
            label_content = f.readlines()
            for label in label_content:
                label = label.strip('\n')
                new_box = [float(label.split(' ')[1]),
                           float(label.split(' ')[2]),
                           float(label.split(' ')[3]),
                           float(label.split(' ')[4])]
                ifSame = False
                for new_label in new_labels:
                    if new_label.split(' ')[0] != label.split(' ')[0]:
                        continue
                    al_box = [float(new_label.split(' ')[1]),
                              float(new_label.split(' ')[2]),
                              float(new_label.split(' ')[3]),
                              float(new_label.split(' ')[4])]
                    if computeIOU(yolov3_to_minmax(al_box), yolov3_to_minmax(new_box)) > 0.5:
                        ifSame = True
                if not ifSame:
                    new_labels.append(label)
            with open(os.path.join(label_out_dir, label_file), 'w') as fo:
                fo.write('\n'.join(new_labels))


if __name__ == '__main__':
    run('/mnt/sdb/data/icra/Rissois_Vegetarianos_labels/',
        '/mnt/sdb/data/icra/Rissois_Vegetarianos_labels_new/')
