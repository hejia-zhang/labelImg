import os
from PIL import Image

WIDTH = 720
HEIGHT = 404


def load_anno(anno_path):
    attributes = {}

    labels = {}

    with open(anno_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            if 'hand' in line:
                continue
            # line = line.replace('spatula', 'spoon')
            splited = line.split()

            if splited[6] == '1':
                continue

            attribute = splited[-1].split('"')[1]
            if attribute not in attributes:
                attributes[attribute] = len(attributes)

            frame = splited[5]
            if frame not in labels:
                labels[frame] = []

            xmin = float(splited[1])
            xmax = float(splited[3])
            ymin = float(splited[2])
            ymax = float(splited[4])
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin
            ratio_box = [str(x_center / WIDTH), str(y_center / HEIGHT), str(width / WIDTH), str(height / HEIGHT)]
            labels[frame].append('{} {}'.format(attributes[attribute], ' '.join(ratio_box)))
            labels[frame].sort()

    for frame in labels:
        labels[frame] = '\n'.join(labels[frame])

    return labels, attributes


def run(image_dir, vatic_anno_path, label_dir, image_out_dir):
    names = os.listdir(image_dir)
    # names.sort()

    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)

    annotations, attributes = load_anno(vatic_anno_path)

    for name in names:
        frame = name.split('.')[0]
        if frame in annotations:
            # write annotations
            with open(os.path.join(label_dir, name.replace('jpg', 'txt')), 'w') as out:
                out.write(annotations[frame])
            img = Image.open(os.path.join(image_dir, name))
            img.save(os.path.join(image_out_dir, name))

    with open(os.path.join(label_dir, 'classes.txt'), 'w') as out:
        for attribute in attributes:
            out.write(attribute + '\n')


if __name__ == '__main__':
    run('/mnt/sdb/data/icra/Massa_Verde/',
        '/mnt/sdb/data/icra/Massa_Verde.txt',
        '/mnt/sdb/data/icra/Massa_Verde_labels/',
        '/mnt/sdb/data/icra/Massa_Verde_images')
