#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import random
from collections import defaultdict
from itertools import izip_longest
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy

from htmresearch.algorithms.apical_tiebreak_temporal_memory import (
    ApicalTiebreakPairMemory,)
from htmresearch.algorithms.location_modules import Superficial2DLocationModule
from htmresearch.frameworks.location.path_integration_union_narrowing import (
    PIUNCorticalColumn,
    PIUNExperiment,)

class PIUNCorticalColumnForVisualRecognition(PIUNCorticalColumn):    
    def __init__(self, locationConfigs,  # noqa: N803
                 L4Overrides=None, bumpType="gaussian"):
        self.bumpType = bumpType

        l4_cell_count = L4Overrides["columnCount"] * L4Overrides["cellsPerColumn"]

        if bumpType == "square":
            self.L6aModules = [
                Superficial2DLocationModule(
                    anchorInputSize=l4_cell_count,
                    **config)
                for config in locationConfigs]
        else:
            raise ValueError("Invalid bumpType", bumpType)

        l4_params = {
            "columnCount": 128,  # Note overriding below
            "cellsPerColumn": 32,
            "basalInputSize": sum(module.numberOfCells()
                                  for module in self.L6aModules)
        }

        if L4Overrides is not None:
            l4_params.update(L4Overrides)

        self.L4 = ApicalTiebreakPairMemory(**l4_params)

    def get_location_copy(self):
        active_cells_list = []

        for module in self.L6aModules:

            active_cells_list.append(module.getActiveCells())

        return active_cells_list

class PIUNExperimentForVisualRecognition(PIUNExperiment):
    def __init__(self, column,
                 sdr1=None, sdr2=None, sdr3=None,
                 num_grid_cells=40 * 256 * 256,
                 num_modules=40,
                 num_classes=10,
                 cell_per_column=32,
                 class_th=0.3,
                 grid_dim=[5],
                 noiseFactor=0,
                 moduleNoiseFactor=0,):

        self.column = column # need not if column are saved
        self.cell_per_column = cell_per_column
        self.class_th = class_th
        self.grid_dim = grid_dim
        self.num_modules = num_modules
        self.num_grid_cells = num_grid_cells
        self.class_weights = np.zeros((num_grid_cells, num_classes))
        self.locationRepresentations = defaultdict(list) # need
        self.sdr1 = sdr1
        self.sdr2 = sdr2
        self.sdr3 = sdr3
        self.locationOnObject = None
        self.maxSettlingTime = 10
        self.monitors = {}
        self.nextMonitorToken = 1
        self.noiseFactor = noiseFactor
        self.moduleNoiseFactor = moduleNoiseFactor
        self.representationSet = set() # need if you wanna recall

    def learnObject(self,  # noqa: N802
                    objdesc1=None, objdesc2=None, objdesc3=None,
                    randomLocation=False,
                    useNoise=False,
                    noisyTrainingTime=1):
        
        self.reset()
        self.column.activateRandomLocation()
        locationsAreUnique = True  # noqa: N806
        all_locations = []
        obj = [objdesc1, objdesc2, objdesc3]
        sdrs = [self.sdr1, self.sdr2, self.sdr3]
        target = int(objdesc1['name'][:objdesc1['name'].index("_")])

        for iFeature, feature in enumerate(obj[0]["features"]):
            self._move(feature, randomLocation=randomLocation, useNoise=useNoise)
            featureSDR = sdrs[0][feature["name"]]  # noqa: N806
            self._sense(featureSDR, learn=True, waitForSettle=False)  # noqa: N806

            locationRepresentation = (  # noqa: N806
                self.column.getSensoryAssociatedLocationRepresentation())
            self.locationRepresentations[(obj[0]["name"],
                                            iFeature)].append(locationRepresentation)

            locationTuple = tuple(locationRepresentation)  # noqa: N806
            locationsAreUnique = (locationsAreUnique  # noqa: N806
                                    and locationTuple not in self.representationSet)
            all_locations.extend(locationRepresentation)
            self.representationSet.add(tuple(locationRepresentation))

        unique_locations = list(set(all_locations))
        self.class_weights[unique_locations, target] += 1
        if objdesc2 is not None:
            for iFeature, feature in enumerate(obj[1]["features"]):
                self._move(feature, randomLocation=randomLocation, useNoise=useNoise)
                featureSDR = sdrs[1][feature["name"]]  # noqa: N806
                self._sense(featureSDR, learn=True, waitForSettle=False)  # noqa: N806

                locationRepresentation = (  # noqa: N806
                    self.column.getSensoryAssociatedLocationRepresentation())
                self.locationRepresentations[(obj[1]["name"],
                                                iFeature)].append(locationRepresentation)

                locationTuple = tuple(locationRepresentation)  # noqa: N806
                locationsAreUnique = (locationsAreUnique  # noqa: N806
                                        and locationTuple not in self.representationSet)
                all_locations.extend(locationRepresentation)
                self.representationSet.add(tuple(locationRepresentation))

            unique_locations = list(set(all_locations))
            self.class_weights[unique_locations, target] += 1
        if objdesc3 is not None:
            for iFeature, feature in enumerate(obj[2]["features"]):
                self._move(feature, randomLocation=randomLocation, useNoise=useNoise)
                featureSDR = sdrs[2][feature["name"]]  # noqa: N806
                self._sense(featureSDR, learn=True, waitForSettle=False)  # noqa: N806

                locationRepresentation = (  # noqa: N806
                    self.column.getSensoryAssociatedLocationRepresentation())
                self.locationRepresentations[(obj[2]["name"],
                                                iFeature)].append(locationRepresentation)

                locationTuple = tuple(locationRepresentation)  # noqa: N806
                locationsAreUnique = (locationsAreUnique  # noqa: N806
                                        and locationTuple not in self.representationSet)
                all_locations.extend(locationRepresentation)
                self.representationSet.add(tuple(locationRepresentation))

            unique_locations = list(set(all_locations))
            self.class_weights[unique_locations, target] += 1

        return locationsAreUnique

    def recallObjectWithRandomMovements(self,  # noqa: C901, N802
                                        objdesc1=None,  # noqa: N803
                                        objdesc2=None,
                                        objdesc3=None,
                                        change=5,
                                        objectwidth=[3,5,7],
                                        cellsPerColumn=32):  # noqa: N803

        self.reset()
        currentStep = 0
        sense_sequence = []
        prediction_sequence = []
        class_dist_rec = []
        data_idx = 0
        obj = [objdesc1, objdesc2, objdesc3]
        sdrs = [self.sdr1, self.sdr2, self.sdr3]
        target = int(obj[data_idx]["name"][0])
        brief_rec = []

        # TOUCH SEQ
        if objectwidth[1] == 0:
            touchSeq = self.get_touchseq(objectwidth[0])
        elif objectwidth[2] == 0:
            ts1 = self.get_touchseq(objectwidth[0])
            ts2 = self.get_touchseq(objectwidth[1])
            touchSeq = ts1[:change[0]] + ts2
        else:
            ts1 = self.get_touchseq(objectwidth[0])
            ts2 = self.get_touchseq(objectwidth[1])
            ts3 = self.get_touchseq(objectwidth[2])
            dif = change[1] - change[0]
            touchSeq = ts1[:change[0]] + ts2[:dif] + ts3

        # recall
        data_idx = 0
        for currentStep, iFeature in enumerate(touchSeq):  # noqa: N806
            if objdesc2 is not None and currentStep == change[0]:
                data_idx = 1
            if objdesc3 is not None and currentStep == change[1]:
                data_idx = 2

            feature = obj[data_idx]["features"][iFeature]
            self._move(feature, randomLocation=False)
            featureSDR = sdrs[data_idx][feature["name"]]
            self._sense(featureSDR, learn=False, waitForSettle=False)

            sense_sequence.append(featureSDR)
            predictedColumns = map(int, list(set(np.floor(  # noqa: N806
                self.column.L4.getBasalPredictedCells() / cellsPerColumn))))
            prediction_sequence.append(predictedColumns)
            representation = self.column.getSensoryAssociatedLocationRepresentation()
            active_loc_vector = np.zeros(self.num_grid_cells)
            active_loc_vector[representation] = 1
            class_dist = np.matmul(active_loc_vector, self.class_weights)
            brief = class_dist[target] / np.sum(class_dist) if np.sum(class_dist) != 0 else 0.
            class_dist_rec.append(class_dist)
            brief_rec.append(brief)

        return brief_rec, sense_sequence, prediction_sequence, touchSeq

    def inferObjectWithRandomMovements(self,  # noqa: C901, N802
                                       objdesc1=None,  # noqa: N803
                                       objdesc2=None,
                                       objdesc3=None,
                                       change=[3],
                                       objectImage=None,  # noqa: N803
                                       cellsPerColumn=32,  # noqa: N803
                                       class_threshold=0.3,
                                       objectwidth=[3,5,7],
                                       infer_regulation=0,
                                       ts_flag=True,
                                       false_motor_information=False,
                                       visualize_predictions_bool=False):

        self.reset()
        currentStep = 0
        sense_sequence = []
        prediction_sequence = []
        class_dist_rec = []
        data_idx = 0
        obj = [objdesc1, objdesc2, objdesc3]
        sdrs = [self.sdr1, self.sdr2, self.sdr3]
        target = int(objdesc1['name'][:objdesc1['name'].index("_")])
        brief_rec = []
        brief_rec_w = []
        entr_rec = []

        # TOUCH SEQ
        #touchSeq = np.arange(objectwidth[0]**2)
        #touchSeq = get_tseq_alternately(objectwidth[1], objectwidth[0])
        touchSeq = [0,35,1,30,2,28,3,25,4,5] if ts_flag is True else [0,35,28,30,25,1,2,3,4,5]
        #tqdm.write(str(touchSeq[:]))
        data_idx_ = [0,1,0,1,0,1,0,1,0,0] if ts_flag is True else [0,1,1,1,1,0,0,0,0,0]
        
        for currentStep, i_feature in enumerate(touchSeq[:25]):
            # CHANGE DATA
            '''
            if objdesc2 is not None and currentStep == change[0]:
                data_idx = 1
            if objdesc3 is not None and currentStep == change[1]:
                data_idx = 2
            if objectwidth[1] != 0:
                data_idx = (currentStep) % 2'''
                #tqdm.write(str(data_idx))
            data_idx = data_idx_[currentStep]
            #tqdm.write(str(data_idx) + '-' + str(i_feature))

            # PREDICTION
            feature = obj[data_idx]["features"][i_feature]
            self._move(feature)
            featureSDR = sdrs[data_idx][feature["name"]]  # noqa: N806
            self._sense(featureSDR, learn=False, waitForSettle=False)

            sense_sequence.append(featureSDR)
            predictedColumns = map(int, list(set(np.floor(  # noqa: N806
                self.column.L4.getBasalPredictedCells() / cellsPerColumn))))
            prediction_sequence.append(predictedColumns)

            representation = self.column.getSensoryAssociatedLocationRepresentation()
            active_loc_vector = np.zeros(self.num_grid_cells)
            active_loc_vector[representation] = 1
            class_dist = np.matmul(active_loc_vector, self.class_weights)
            entr = entropy(class_dist,)
            entr_rec.append(entr)
            #tqdm.write(str(entr))
            brief = class_dist[target] / np.sum(class_dist) if np.sum(class_dist) != 0 else 0.
            class_dist_rec.append(class_dist)
            brief_rec.append(brief)

            wrong_target = 10 if target == 11 else 11
            brief_w = class_dist[wrong_target] / np.sum(class_dist) if np.sum(class_dist) != 0 else 0.
            brief_rec_w.append(brief_w)

        return brief_rec, sense_sequence, prediction_sequence, touchSeq, brief_rec_w, entr_rec

'''
if objectwidth[1] == 0:
            touchSeq = get_touchseq_fix(objectwidth[0])
elif objectwidth[2] == 0:
    ts1 = get_touchseq_fix(objectwidth[0])
    ts2 = get_touchseq_fix(objectwidth[1], order=2)
    touchSeq = np.concatenate([ts1[:5], ts2])'''

def get_touchseq_fix(width=5, fix=[], order=1):
    if width == 5:
        fix = [24,23,19,18,20,21,15,16]
    elif width == 7:
        fix = [41,40,34,33,35,36,28,29]
    elif width == 3:
        fix = [6,8]
    
    if order == 2:
        return fix
    
    ts = np.random.permutation(width**2)
    mask_ts = ts[~np.isin(ts, fix)]
    touchseq = np.concatenate([mask_ts[:5], fix])
    return touchseq

def get_tseq_alternately(width1, width2):
    width2 = width1 if width2 == 0 else width2

    if width2 == 5:
        feat = [24,23,19,18,20,21,15,16]
    elif width2 == 7:
        feat = [41,40,34,33,35,36,28,29]
    elif width2 == 4:
        feat = [15,14,11,12,13,8]
    elif width2 == 3:
        feat = [6,8]
    elif width2 == 6:
        feat = [0]#[30,31,24,25,35,34,29,28]#[35,34,29,28,30,31,24,25]

    if width1 == 5:
        msk = [24,23,19,18,20,21,15,16]
    elif width1 == 4:
        msk = [15,14,11,12,13,8]
    elif width1 == 7:
        msk = [41,40,34,33,35,36,28,29]
    elif width1 == 3:
        msk = [6,8]
    elif width1 == 6:
        msk = [30,31,24,25,35,34,29,28]#[35,34,29,28,30,31,24,25]
    
    ts = np.random.permutation(width1**2)
    mask_ts = ts[~np.isin(ts, msk)]
    touchseq = [item for pair in izip_longest(feat, mask_ts) for item in pair if item is not None]
    return touchseq

def get_allfixed(width1=5, width2=5):
    if width1 == 3:
        feat = [6,8]
    elif width1 == 4:
        feat = [15,14,11,12,13,8]
    elif width1 == 5:
        feat = [24,23,19,18,20,21,15,16]
    elif width1 == 6:
        feat = [35,34,29,28,30,31,24,25]
    elif width1 == 7:
        feat = [41,40,34,33,35,36,28,29]
    
    if width2 == 3:
        nfeat = [1,2,3,4,5,7]
    elif width2 == 4:
        nfeat = [1,2,3,4,5,6,7,9,10]
    
    ts = np.random.permutation(width1**2)
    mask_ts = ts[~np.isin(ts, nfeat)]
    touchseq = [item for pair in izip_longest(mask_ts, feat) for item in pair if item is not None]
    return touchseq

'''
        if objectwidth[1] == 0:
            touchSeq = self.get_touchseq(objectwidth[0])
        elif objectwidth[2] == 0:
            ts1 = self.get_touchseq(objectwidth[0])
            ts2 = self.get_touchseq(objectwidth[1])
            touchSeq = ts1[:change[0]] + ts2
        else:
            ts1 = self.get_touchseq(objectwidth[0])
            ts2 = self.get_touchseq(objectwidth[1])
            ts3 = self.get_touchseq(objectwidth[2])
            dif = change[1] - change[0]
            touchSeq = ts1[:change[0]] + ts2[:dif] + ts3

fixed touch seq
        ts1 = np.random.permutation(25)
        pops1 = ts1[np.isin(ts1, [15,16,18,19,20,21,23,24])]
        ts2 = np.random.permutation(49)
        pops2 = ts2[np.isin(ts2, [28,29,30,32,33,34,35,36,37,39,40,41,42,43,44,46,47,48])]
        if objectwidth == [5,7,0]:
            touchSeq = np.concatenate([ts1[:10], pops2])
            change[0] = 10
        elif objectwidth == [7,5,0]:
            touchSeq = np.concatenate([ts2[:10], pops1])
            change[0] = 10
        elif objectwidth == [5,0,0]:
            touchSeq = np.concatenate([ts1[:10], pops1])
        elif objectwidth == [7,0,0]:
            touchSeq = np.concatenate([ts2[:10], pops2])

def get_touchseq(self, objectwidth):
        n = 49 // (objectwidth**2) + 1
        ts = []
        for i in range(n):
            ts.extend(np.arange(objectwidth**2))
        return ts
#'''