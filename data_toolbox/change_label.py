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


def add(add_table, new_labels, label_content):
    labels = []
    for label in label_content:
        if label[0] == add_table['container_type']:
            labels.append(label.strip('\n'))
    if len(labels) == len(add_table['contained_type']):
        # sort labels based on center of box
        labels.sort(key=lambda label: yolov3_to_minmax(
            [label.split(' ')[1], label.split(' ')[2], label.split(' ')[3], label.split(' ')[4]])[0] + yolov3_to_minmax(
            [label.split(' ')[1], label.split(' ')[2], label.split(' ')[3], label.split(' ')[4]])[2])
        # after sort labels, add new labels
        for i, _ in enumerate(labels):
            new_labels.append('{} {} {} {} {}'.format(add_table['contained_type'][i], labels[i].split(' ')[1],
                                                      labels[i].split(' ')[2], float(labels[i].split(' ')[3]) * 0.8,
                                                      float(labels[i].split(' ')[4]) * 0.8))


def run(change_table_lst, label_dir, label_out_dir):
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

                for change_table in change_table_lst:
                    if int(label_file.split('.')[0]) in change_table['range']:
                        if change_table['wrong_type'] == label.split(' ')[0]:
                            label = '{} {} {} {} {}'.format(change_table['right_type'], label.split(' ')[1],
                                                            label.split(' ')[2], label.split(' ')[3],
                                                            label.split(' ')[4])

                new_labels.append(label)

            with open(os.path.join(label_out_dir, label_file), 'w') as fo:
                fo.write('\n'.join(new_labels))


if __name__ == '__main__':
    change_table_lst = [{'wrong_type': '16', 'range': range(2662, 2722), 'right_type': '9'}]
    run(change_table_lst,
        '/mnt/sdb/data/icra/Massa_Verde_labels/',
        '/mnt/sdb/data/icra/Massa_Verde_labels_new/')
