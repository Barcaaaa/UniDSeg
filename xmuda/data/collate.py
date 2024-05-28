import torch
from functools import partial


def collate_scn_base(input_dict_list, output_orig, output_image=True):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :param output_orig: whether to output original point cloud/labels/indices
    :param output_image: whether to output images
    :return: Collated data batch as dict
    """
    locs = []
    feats = []
    labels = []
    depths = []
    dense_depths = []
    intensity_maps = []
    ori_full_imgs = []
    img_boundaries = []
    img_objects = []

    if output_image:
        imgs = []
        # imgs_no_jitter = []
        img_idxs = []

    if output_orig:
        orig_seg_label = []
        orig_points_idx = []

    output_pselab = 'pseudo_label_2d' in input_dict_list[0].keys()
    if output_pselab:
        pseudo_label_2d = []
        pseudo_label_3d = []
        pseudo_label_ensemble = []

    for idx, input_dict in enumerate(input_dict_list):
        coords = torch.from_numpy(input_dict['coords'])
        batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
        locs.append(torch.cat([coords, batch_idxs], 1))
        feats.append(torch.from_numpy(input_dict['feats']))
        if 'seg_label' in input_dict.keys():
            labels.append(torch.from_numpy(input_dict['seg_label']))
        if output_image:
            imgs.append(torch.from_numpy(input_dict['img']))
            # imgs_no_jitter.append(torch.from_numpy(input_dict['img_no_jitter']))
            img_idxs.append(input_dict['img_indices'])
            if 'ori_full_img' in input_dict.keys():
                ori_full_imgs.append(torch.from_numpy(input_dict["ori_full_img"]))
            if 'depth' in input_dict.keys():
                depths.append(torch.from_numpy(input_dict["depth"]))
            if 'dense_depth' in input_dict.keys():
                dense_depths.append(torch.from_numpy(input_dict["dense_depth"]))
            if 'intensity_map' in input_dict.keys():
                intensity_maps.append(torch.from_numpy(input_dict["intensity_map"]))
            if 'img_boundary' in input_dict.keys():
                img_boundaries.append(torch.from_numpy(input_dict["img_boundary"]))
            if 'img_object' in input_dict.keys():
                img_objects.append(torch.from_numpy(input_dict["img_object"]))

        if output_orig:
            orig_seg_label.append(input_dict['orig_seg_label'])
            orig_points_idx.append(input_dict['orig_points_idx'])
        if output_pselab:
            pseudo_label_2d.append(torch.from_numpy(input_dict['pseudo_label_2d']))
            if input_dict['pseudo_label_3d'] is not None:
                pseudo_label_3d.append(torch.from_numpy(input_dict['pseudo_label_3d']))
            if input_dict['pseudo_label_ensemble'] is not None:
                pseudo_label_ensemble.append(torch.from_numpy(input_dict['pseudo_label_ensemble']))

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    out_dict = {'x': [locs, feats]}
    if labels:
        labels = torch.cat(labels, 0)
        out_dict['seg_label'] = labels
    if output_image:
        out_dict['img'] = torch.stack(imgs)
        # out_dict['img_no_jitter'] = torch.stack(imgs_no_jitter)
        out_dict['img_indices'] = img_idxs
        if 'ori_full_img' in input_dict.keys():
            out_dict["ori_full_img"] = torch.stack(ori_full_imgs)
        if 'depth' in input_dict.keys():
            out_dict["depth"] = torch.stack(depths)
        if 'dense_depth' in input_dict.keys():
            out_dict["dense_depth"] = torch.stack(dense_depths)
        if 'intensity_map' in input_dict.keys():
            out_dict["intensity_map"] = torch.stack(intensity_maps)
        if 'img_boundary' in input_dict.keys():
            out_dict['img_boundary'] = torch.stack(img_boundaries)
        if 'img_object' in input_dict.keys():
            out_dict['img_object'] = torch.stack(img_objects)
    if output_orig:
        out_dict['orig_seg_label'] = orig_seg_label
        out_dict['orig_points_idx'] = orig_points_idx
    if output_pselab:
        out_dict['pseudo_label_2d'] = torch.cat(pseudo_label_2d, 0)
        out_dict['pseudo_label_3d'] = torch.cat(pseudo_label_3d, 0) if pseudo_label_3d else pseudo_label_3d
        out_dict['pseudo_label_ensemble'] = torch.cat(pseudo_label_ensemble, 0) if pseudo_label_ensemble else pseudo_label_ensemble
    return out_dict


def get_collate_scn(is_train):
    return partial(collate_scn_base,
                   output_orig=not is_train,
                   )
