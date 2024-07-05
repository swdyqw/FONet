import imp
import re

import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class GTOTDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.GTOT_path
        self.sequence_list = self._get_sequence_list(self.base_path)
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls = self.sequence_list[i]
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name
        anno_path = '{}/{}/init.txt'.format(self.base_path, class_name)

        ground_truth_rect = load_text(str(anno_path), delimiter='\t', dtype=np.float64)
        # print(ground_truth_rect.shape[0])
        # occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = np.zeros((ground_truth_rect.shape[0], ), dtype=np.float64)

        # out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        out_of_view = np.zeros([ground_truth_rect.shape[0], ], np.float64)

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/v'.format(self.base_path, class_name)
        frames_path_i = '{}/{}/i'.format(self.base_path, class_name)

        pattern = re.compile(r'\d+')

        list1 = os.listdir(frames_path)
        list1.sort()
        frames_list = [os.path.join(frames_path, x) for x in list1]

        list2 = os.listdir(frames_path_i)
        list2.sort()
        frames_list_i = [os.path.join(frames_path_i, x) for x in list2]
        target_class = class_name
        return Sequence(sequence_name, frames_list, frames_list_i, 'GTOT', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)
 
    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, path):
        sequence_list = os.listdir(path)
        return sequence_list
