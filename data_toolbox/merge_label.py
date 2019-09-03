import os
from PIL import Image

WIDTH = 720
HEIGHT = 404

# videos = ['Arroz_do_Mar', 'bol', 'Couve_Lombarda',
#           'da_Prima', 'limao', 'Massa_Verde']
videos = ['try', 'Lasanha', 'limao']

dir_path = '/home/icaros-root/Desktop/STORE'

imgs_out = '/home/icaros-root/Desktop/STORE/images'
labels_out = '/home/icaros-root/Desktop/STORE/labels'


def merge_classes():
    classes_set = {}
    class_num = 0
    for video in videos:
        class_path = os.path.join(dir_path, video, '{}_labels'.format(video), 'classes.txt')
        with open(class_path, 'r') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                if line not in classes_set:
                    classes_set[line] = class_num
                    class_num += 1
    return classes_set


def run():
    img_num = 0
    classes_set = merge_classes()
    for video in videos:
        classes_set_temp = {}
        class_num_temp = 0
        class_path = os.path.join(dir_path, video, '{}_labels'.format(video), 'classes.txt')
        visited = {}
        with open(class_path, 'r') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                if line not in visited:
                    classes_set_temp[class_num_temp] = line
                    class_num_temp += 1
                    visited[line] = True
        img_dir = os.path.join(dir_path, video, '{}_images'.format(video))
        label_dir = os.path.join(dir_path, video, '{}_labels'.format(video))
        names = os.listdir(img_dir)

        for name in names:
            img = Image.open(os.path.join(img_dir, name))
            img.save(os.path.join(imgs_out, '{}.jpg'.format(img_num)))

            new_lines = []
            with open(os.path.join(label_dir, name.replace('jpg', 'txt')), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split('\n')[0]
                    class_idx_str = line.split()[0]
                    class_name = classes_set_temp[int(class_idx_str)]
                    new_class_idx = classes_set[class_name]
                    new_line = [str(new_class_idx)] + line.split()[1:]
                    new_line_str = ' '.join(new_line)
                    new_lines.append(new_line_str)

            with open(os.path.join(labels_out, '{}.txt'.format(img_num)), 'w') as out:
                for line in new_lines:
                    out.write(line + '\n')

            img_num += 1

    with open(os.path.join(labels_out, 'classes.txt'), 'w') as out:
        for class_name in classes_set:
            out.write(class_name + '\n')


if __name__ == '__main__':
    run()
