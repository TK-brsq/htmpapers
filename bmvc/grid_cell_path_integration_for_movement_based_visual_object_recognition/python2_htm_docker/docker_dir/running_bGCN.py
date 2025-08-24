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

import json
import math
import os
import random
from datetime import datetime
import pytz
from tqdm import tqdm
import numpy as np
from bGCN import (
    PIUNCorticalColumnForVisualRecognition,
    PIUNExperimentForVisualRecognition,)
from generate_data import generate_image_objects

# ========== HYPER PARAMETERS ==========
NUM_CLASSES = 12
VARIETY = 1
NUM_SEED = 11
NUM_SAMPLES_PER_CLASS = 20
NUM_TEST_OBJECTS_PER_CLASS = 20
OBJECT_WIDTHS = [[3,6,0],[3,6,0], [7,0,0]] #
CHANGE = [[5,0],[5,0],[10,0]] #
TS_FLAG = [True, False]
# ===========================================
EVAL_ON_TRAINING_DATA_BOOL = False
INFER_REGULATION = 0
THRESHOLD_LIST = [16]
CLASS_THRESHOLDS_LIST = [0.3]
DATASET = "mnist"  # Options are "mnist" or "fashion_mnist"
SEED1 = 11
SEED2 = 12
BLOCKED_TRAINING_BOOL = True
BLOCK_SPLIT = None
FALSE_MOTOR_INFORMATION = False
VISUALIZE_PREDICTIONS_BOOL = False
FIXED_TOUCH_SEQUENCE = None
if FIXED_TOUCH_SEQUENCE is not None:
    random.shuffle(FIXED_TOUCH_SEQUENCE)
if EVAL_ON_TRAINING_DATA_BOOL:
    assert NUM_SAMPLES_PER_CLASS == NUM_TEST_OBJECTS_PER_CLASS, \
        "If evaluating recall, ensure the number" \
        "of training and test objects is the same"
FLIP_BITS_LIST = [0]

def object_learning_and_inference(
        EVAL_ON_TRAINING_DATA_BOOL,  # noqa: N803
        locationModuleWidth,
        feature_columns_to_grid_cells_threshold,
        grid_cells_to_feature_columns_threshold,
        class_threshold,
        bumpType,
        cellCoordinateOffsets,
        cellsPerColumn,
        activeColumnCount,
        columnCount,
        objectWidths,
        numModules,
        seed1,
        seed2,
        anchoringMethod,
        change,
        ts_flag):

    np.random.seed(seed1)
    random.seed(seed2)

    # Data Loader
    train_sdr1, train_objects1, object_images = generate_image_objects(
        DATASET, NUM_SAMPLES_PER_CLASS, objectWidths[0],
        locationModuleWidth, data_set_section="SDR_classifiers_training", num_classes=NUM_CLASSES, seed=seed1)
    train_sdr2, train_objects2, _ = generate_image_objects(
        DATASET, NUM_SAMPLES_PER_CLASS, objectWidths[1],
        locationModuleWidth, data_set_section="SDR_classifiers_training", num_classes=NUM_CLASSES, seed=seed1)
    train_sdr3, train_objects3, _ = generate_image_objects(
        DATASET, NUM_SAMPLES_PER_CLASS, objectWidths[2],
        locationModuleWidth, data_set_section="SDR_classifiers_training", num_classes=NUM_CLASSES, seed=seed1)
    if EVAL_ON_TRAINING_DATA_BOOL:
        test_sdr1, test_objects1, object_images = train_sdr1, train_objects1, object_images
        test_sdr2, test_objects2 = train_sdr2, train_objects2
        test_sdr3, test_objects3 = train_sdr3, train_objects3
    else:
        test_sdr1, test_objects1, object_images1 = generate_image_objects(
            DATASET, NUM_TEST_OBJECTS_PER_CLASS, objectWidths[0],
            locationModuleWidth, data_set_section="SDR_classifiers_testing", num_classes=NUM_CLASSES, seed=seed1)
        test_sdr2, test_objects2, _ = generate_image_objects(
            DATASET, NUM_TEST_OBJECTS_PER_CLASS, objectWidths[1],
            locationModuleWidth, data_set_section="SDR_classifiers_testing", num_classes=NUM_CLASSES, seed=seed1)
        test_sdr3, test_objects3, _ = generate_image_objects(
            DATASET, NUM_TEST_OBJECTS_PER_CLASS, objectWidths[2],
            locationModuleWidth, data_set_section="SDR_classifiers_testing", num_classes=NUM_CLASSES, seed=seed1)
    #region My Custom Section
    locationConfigs = []  # noqa: N806
    perModRange = float((90.0 if bumpType == "square" else 60.0)  # noqa: N806
                        / float(numModules))
    
    for i in range(numModules):
        orientation = (float(i) * perModRange) + (perModRange / 2.0)
        config = {
            "cellsPerAxis": locationModuleWidth,
            "scale": 40.0,
            "orientation": np.radians(orientation),
            "activationThreshold": feature_columns_to_grid_cells_threshold,
            "initialPermanence": 1.0,
            "connectedPermanence": 0.5,
            "learningThreshold": feature_columns_to_grid_cells_threshold,
            "sampleSize": -1,    # during learning, setting this to -1 means max new synapses = len(activeInput)
            "permanenceIncrement": 0.1,
            "permanenceDecrement": 0.0,
            "cellCoordinateOffsets": cellCoordinateOffsets,
            "anchoringMethod": anchoringMethod}

        locationConfigs.append(config)

    l4Overrides = {  # noqa: N806
        "initialPermanence": 1.0,
        "activationThreshold": int(
            math.ceil(grid_cells_to_feature_columns_threshold * numModules)),
        "reducedBasalThreshold": int(
            math.ceil(grid_cells_to_feature_columns_threshold * numModules)),
        "minThreshold": numModules,
        "sampleSize": numModules,
        "cellsPerColumn": cellsPerColumn,
        "columnCount": columnCount}

    allLocationsAreUnique = None  # noqa: N806
    column = PIUNCorticalColumnForVisualRecognition(
        locationConfigs, L4Overrides=l4Overrides, bumpType=bumpType)
    ColumnPlusNet = PIUNExperimentForVisualRecognition(  # noqa: N806
        column, train_sdr1, train_sdr2, train_sdr3,
        numModules*locationModuleWidth**2,
        numModules, num_classes=12)
    # endregion

    # Learning
    currentLocsUnique = True  # noqa: N806
    for obj1, obj2, obj3 in tqdm(zip(train_objects1, train_objects2, train_objects3)):  # noqa: N806
        objLocsUnique = ColumnPlusNet.learnObject(obj1, obj2, obj3,)
        currentLocsUnique = currentLocsUnique and objLocsUnique

    # Evaluation
    ColumnPlusNet.sdr1 = test_sdr1
    ColumnPlusNet.sdr2 = test_sdr2
    ColumnPlusNet.sdr3 = test_sdr3

    if EVAL_ON_TRAINING_DATA_BOOL:
        brief_rec_rec = []
        for objdesc1, objdesc2, objdesc3 in tqdm(zip(test_objects1, test_objects2, test_objects3)):  # noqa: N806
            brief_rec, sense_rec, AC_rec, touchseq = ColumnPlusNet.recallObjectWithRandomMovements(
                    objdesc1, objdesc2, objdesc3, change=change)

        result = {
            "brief": brief_rec_rec,
            "touchseq": touchseq,
            "sense": sense_rec,
            "AC": AC_rec,
            "change": change,
            "data": objectWidths,
            "recallorinfer": 'recall',
        }
    else:
        brief_rec_rec = []
        brief_w_rec_rec = []
        target_rec = []
        touchseq_rec = []
        sense_rec_rec = []
        AC_rec_rec = []
        entropy_rec = []
        for objdesc1, objdesc2, objdesc3 in tqdm(zip(test_objects1, test_objects2, test_objects3)):
            l = int(objdesc1['name'][:objdesc1['name'].index("_")])
            target_rec.append(l)
            brief_rec, sense_rec, AC_rec, touchseq, brief_w_rec, entr = ColumnPlusNet.inferObjectWithRandomMovements(
                    objdesc1, objdesc2, objdesc3,
                    change=change,
                    objectImage=None,
                    cellsPerColumn=cellsPerColumn,
                    class_threshold=class_threshold,
                    objectwidth=objectWidths,
                    ts_flag=ts_flag)
            brief_rec_rec.append(brief_rec)
            brief_w_rec_rec.append(brief_w_rec)
            touchseq_rec.append(touchseq)
            sense_rec_rec.append(sense_rec)
            AC_rec_rec.append(AC_rec)
            entropy_rec.append(entr)

        result = {
            "brief": brief_rec_rec,
            "touchseq": touchseq_rec,
            "sense": sense_rec_rec,
            "AC": AC_rec_rec,
            "change": change,
            "data": objectWidths,
            "recallorinfer": 'infer',
            "target": target_rec,
            "brief_w": brief_w_rec_rec,
            "entropy": entropy_rec,
        }
    return result


def dir_setup():
    if os.path.exists("misclassified/") is False:
        try:
            os.mkdir("misclassified/")
        except OSError:
            pass

    if os.path.exists("correctly_classified/") is False:
        try:
            os.mkdir("correctly_classified/")
        except OSError:
            pass

    if os.path.exists("results/") is False:
        try:
            os.mkdir("results/")
        except OSError:
            pass

    if os.path.exists("prediction_data/") is False:
        try:
            os.mkdir("prediction_data/")
        except OSError:
            pass


if __name__ == "__main__":
    dir_setup()

    cellCoordinateOffsets = tuple([i * (0.998) + 0.001 for i in range(2)])

    for i in range(VARIETY):
        jst = pytz.timezone("Asia/Tokyo")
        japan_time = datetime.now(jst)
        now_ = japan_time.strftime("%m-%d-%H-%M")
        record_result = []
        for iter in range(NUM_SEED):
            print("Run : ", iter+1)
            result = object_learning_and_inference(
                EVAL_ON_TRAINING_DATA_BOOL=EVAL_ON_TRAINING_DATA_BOOL,
                locationModuleWidth=256,
                feature_columns_to_grid_cells_threshold=THRESHOLD_LIST,
                grid_cells_to_feature_columns_threshold=0.9,
                class_threshold=CLASS_THRESHOLDS_LIST,
                bumpType="square",
                cellsPerColumn=32,
                columnCount=128,
                activeColumnCount=29,
                cellCoordinateOffsets=cellCoordinateOffsets,
                objectWidths=OBJECT_WIDTHS[i],
                numModules=40, 
                seed1=SEED1*(iter+1),
                seed2=SEED2*(iter+1),
                anchoringMethod="corners",
                change=CHANGE[i],
                ts_flag=TS_FLAG[i])
            record_result.append(result)

        np.save("results/" + now_ + "-" + str(OBJECT_WIDTHS[i]) + ".npy", record_result)
        print("----- Result were saved -----\n")
