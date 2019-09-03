from copy import deepcopy
import os
import os.path as osp

from collections import OrderedDict
import numpy as np


def obj_tracking(labels_lst, max_disappered, obj_type):
    objects = OrderedDict()
    disappeared = OrderedDict()
    next_object_id = 0

    def register(anno):
        # When registering an object we use the next available object
        # ID to store the centroid
        nonlocal next_object_id
        objects[next_object_id] = anno
        disappeared[next_object_id] = 0
        next_object_id += 1

    def deregister(object_id):
        # To deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del objects[object_id]
        del disappeared[object_id]

    def update(anno_lst):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(anno_lst) == 0:
            # loop over all existing tracked objects and mark them
            # as disappeared
            for object_id in list(disappeared.keys()):
                disappeared[object_id] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if disappeared[object_id] > max_disappered:
                    deregister(object_id)
            # return early as there are no centroids or tracking info
            # to update
            return
        # input_hands = []
        # for person in anno:
        #     for hand in anno[person]:
        #         input_hands.append({'box': anno[person][hand]['box'],
        #                             'hand_name': '{}_{}'.format(person, hand)})
        # if we are currently not tracking any object take the input
        # centroids and register each of them
        if len(objects) == 0:
            for anno in anno_lst:
                register(anno)
        # otherwise, we are currently tracking objects so we need to
        # try to match the input centroids to existing objects
        else:
            # grab the set of object IDs and corresponding centroids
            object_ids = list(objects.keys())
            rows = set(object_ids)
            cols = set(range(len(anno_lst)))
            used_rows = set()
            used_cols = set()
            for object_id in object_ids:
                for i, anno in enumerate(anno_lst):
                    if anno['type'] == objects[object_id]['type']:
                        objects[object_id] = anno
                        disappeared[object_id] = 0
                        used_rows.add(object_id)
                        used_cols.add(i)
            unused_rows = rows.difference(used_rows)
            unused_cols = cols.difference(used_cols)
            for row in unused_rows:
                object_id = row
                disappeared[object_id] += 1
                if disappeared[object_id] > max_disappered:
                    deregister(object_id)
            # we need to register each new input centroid as a trackable object
            for col in unused_cols:
                register(anno_lst[col])
            # if len(rows) > len(cols):
            #     for row in unused_rows:
            #         object_id = row
            #         disappeared[object_id] += 1
            #
            #         if disappeared[object_id] > max_disappered:
            #             deregister(object_id)
            # elif len(rows) < len(cols):
            #     # we need to register each new input centroid as a trackable object
            #     for col in unused_cols:
            #         register(input_hands[col])

    objects_lst = []
    disappeared_lst = []
    for label_path in labels_lst:
        anno_lst = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.split(' ')[0] != obj_type:
                    continue
                anno_dict = {}
                line = line.strip('\n')
                anno_dict['type'] = line.split(' ')[0]
                anno_dict['box'] = np.array([float(line.split(' ')[1]),
                                             float(line.split(' ')[2]),
                                             float(line.split(' ')[3]),
                                             float(line.split(' ')[4])])
                anno_lst.append(anno_dict)
        update(anno_lst)
        objects_lst.append(deepcopy(objects))
        disappeared_lst.append(deepcopy(disappeared))
    return objects_lst, disappeared_lst


def linear_interpolating(objects_lst, disappeared_lst):
    interpolated_objects_lst = deepcopy(objects_lst)
    # collect all unique object ids
    object_id_set = set()
    for objects in objects_lst:
        for object_id in objects.keys():
            object_id_set.add(object_id)
    # interpolate for each object id
    for object_id in object_id_set:
        start_idx = -1
        end_idx = -1
        for i, objects in enumerate(objects_lst):
            if object_id in objects:
                if start_idx == -1:
                    start_idx = i
                    end_idx = i
                else:
                    end_idx = i
            # else:
            # if object id is not in objects
            # it will never appear again
            # break
        small_start_idx = start_idx
        for i in range(start_idx + 1, end_idx):
            if disappeared_lst[i][object_id] == 0:
                small_start_idx = i
                continue
            else:
                if disappeared_lst[i + 1][object_id] > 0:
                    continue
                else:
                    small_end_idx = i + 1
                    # interpolate between small_start_idx and small_end_idx
                    for idx in range(small_start_idx + 1, small_end_idx):
                        interpolated_objects_lst[idx][object_id]['box'] = \
                            interpolated_objects_lst[small_start_idx][object_id]['box'] + \
                            (interpolated_objects_lst[small_end_idx][object_id]['box'] -
                             interpolated_objects_lst[small_start_idx][object_id]['box']) / \
                            (small_end_idx - small_start_idx) * (idx - small_start_idx)
    return interpolated_objects_lst


def label_paths_from_label_dir(label_dir: str):
    label_names = os.listdir(label_dir)
    _, ext = osp.splitext(label_names[0])
    label_idxes = []
    for label_name in label_names:
        if 'class' in label_name:
            continue
        label_name = label_name.replace(ext, '')
        label_idx = int(label_name)
        label_idxes.append(label_idx)
    label_idxes.sort()
    label_path_lst = []
    for idx in label_idxes:
        label_path_lst.append(osp.join(label_dir, '{}{}'.format(idx, ext)))
    return label_path_lst


def run(interpolate_table, label_dir, label_out_dir):
    os.makedirs(label_out_dir, exist_ok=True)

    labels = label_paths_from_label_dir(label_dir)

    obj_lst, disappeared_lst = obj_tracking(labels[interpolate_table['range'][0]:interpolate_table['range'][-1] + 1],
                                            interpolate_table['max_dis'],
                                            interpolate_table['type'])

    interpolated_objects_lst = linear_interpolating(obj_lst, disappeared_lst)

    for i, label_file in enumerate(labels):

        if 'class' in label_file:
            continue
        new_labels = []

        with open(os.path.join(label_dir, label_file), 'r') as f:
            label_content = f.readlines()
            for label in label_content:
                label = label.strip('\n')
                new_labels.append(label)
            if i in interpolate_table['range']:
                start = i - interpolate_table['range'][0]
                if disappeared_lst[start][0] > 0:
                    new_labels.append('{} {} {} {} {}'.format(interpolated_objects_lst[start][0]['type'],
                                                              interpolated_objects_lst[start][0]['box'][0],
                                                              interpolated_objects_lst[start][0]['box'][1],
                                                              interpolated_objects_lst[start][0]['box'][2],
                                                              interpolated_objects_lst[start][0]['box'][3], ))

        with open(os.path.join(label_out_dir, osp.basename(label_file)), 'w') as fo:
            fo.write('\n'.join(new_labels))


if __name__ == '__main__':
    interpolate_table = {'type': '20', 'range': range(6807, 6900), 'max_dis': 10}

    run(interpolate_table,
        '/mnt/sdb/data/icra/Massa_Verde_labels/',
        '/mnt/sdb/data/icra/Massa_Verde_labels_new/')
