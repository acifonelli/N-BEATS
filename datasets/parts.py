# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Parts Dataset
"""
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patoolib
from typing import Tuple

from common.http_utils import download, url_file_name
from common.settings import DATASETS_PATH

DATASET_URL = 'https://github.com/robjhyndman/expsmooth/blob/master/data/carparts.rda'

DATASET_PATH = os.path.join(DATASETS_PATH, 'parts')
DATASET_FILE_PATH = os.path.join(DATASET_PATH, url_file_name(DATASET_URL))


@dataclass()
class PartsMeta:
    seasonal_pattern = 'Monthly'
    horizon = 12
    frequency = 12


@dataclass()
class PartsDataset:
    ids: np.ndarray
    groups: np.ndarray
    values: np.ndarray

    @staticmethod
    def load() -> 'PartsDataset':
        """
        Load Parts dataset from cache.

        """
        values = []

        dataset = pd.read_csv(os.path.join(DATASET_PATH, 'carparts.csv'),
                            header=0, delimiter=",")
        pivoted = dataset.pivot(index="Id", columns="Date", values="Qty")
        
        ids = pivoted.index.to_numpy()
        groups = np.array(['Spare'] * pivoted.shape[0])

        # ===> Array[Array] <===
        # values = np.empty(pivoted.shape[0], dtype=np.ndarray)
        # for row in enumerate(pivoted.iterrows()):
        #     idx, data = row
        #     values[idx] = data[1].to_numpy()
        
        # ===> Matrix <===
        values = pivoted.to_numpy()

        return PartsDataset(ids=ids,
                            groups=groups,
                            values=values)
    
    def split(self, cut_point: int) -> Tuple['PartsDataset', 'PartsDataset']:
        """
        Split dataset by cut point.

        :param cut_point: Cut index.
        :return: Two parts of dataset: the left part contains all points before the cut point
        (training) and the right part contains all datpoints on and after the cut point (testing).
        """
        return PartsDataset(ids=self.ids,
                            groups=self.groups[:cut_point],
                            values=self.values[:, :cut_point]), \
               PartsDataset(ids=self.ids,
                            groups=self.groups[cut_point:],
                            values=self.values[:, cut_point:])

    # @staticmethod
    # def download():
    #     """
    #     Download Parts dataset.
    #     """
    #     if os.path.isdir(DATASET_PATH):
    #         logging.info(f'skip: {DATASET_PATH} directory already exists.')
    #         return
    #     download(DATASET_URL, DATASET_FILE_PATH)
    #     patoolib.extract_archive(DATASET_FILE_PATH, outdir=DATASET_PATH)

    # def to_hp_search_training_subset(self):
    #     return PartsDataset(ids=self.ids,
    #                           groups=self.groups,
    #                           horizons=self.horizons,
    #                           values=np.array([v[:-self.horizons[i]] for i, v in enumerate(self.values)]))
