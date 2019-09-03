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


def computeSize(label: str):
    minmax = yolov3_to_minmax([label.split(' ')[1], label.split(' ')[2], label.split(' ')[3], label.split(' ')[4]])
    return (minmax[2] - minmax[0]) * (minmax[3] - minmax[1])


def add(add_table, new_labels, label_content):
    labels = []
    for label in label_content:
        if label.split(' ')[0] == add_table['container_type']:
            labels.append(label.strip('\n'))
    # sort labels based on size of box
    labels.sort(key=lambda label: yolov3_to_minmax(
            [label.split(' ')[1], label.split(' ')[2], label.split(' ')[3], label.split(' ')[4]])[1] + yolov3_to_minmax(
            [label.split(' ')[1], label.split(' ')[2], label.split(' ')[3], label.split(' ')[4]])[3])
    for i in range(len(add_table['contained_type'])):
        if i >= len(labels):
            break
        new_labels.append('{} {} {} {} {}'.format(add_table['contained_type'][i], labels[i].split(' ')[1],
                                                  labels[i].split(' ')[2], float(labels[i].split(' ')[3]) * 0.8,
                                                  float(labels[i].split(' ')[4]) * 0.8))


def run(add_table_lst, label_dir, label_out_dir):
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
                new_labels.append(label)
            # based on new_labels, add contained type
            for add_table in add_table_lst:
                if int(label_file.split('.')[0]) in add_table['range']:
                    add(add_table, new_labels, label_content)

            with open(os.path.join(label_out_dir, label_file), 'w') as fo:
                fo.write('\n'.join(new_labels))


if __name__ == '__main__':
    add_table_lst = [{'container_type': '20', 'range': range(6807, 6900), 'contained_type': ['26']}]
    run(add_table_lst,
        '/mnt/sdb/data/icra/Massa_Verde_labels/',
        '/mnt/sdb/data/icra/Massa_Verde_labels_new/')
