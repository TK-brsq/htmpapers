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
import numpy as np

def generate_image_objects(
        data_set, num_samples_per_class, objectWidth,  # noqa: N803
        locationModuleWidth,  # noqa: N803
        data_set_section="SDR_classifiers_training", block_split=None, num_classes=10, seed=0,):
    
    np.random.seed(seed)

    if objectWidth == 0:
        sdr = None
        desc = [None] * num_samples_per_class * num_classes
        img = None
        return sdr, desc, img
    
    name = 'dataset' + str(num_classes) + '-' + str(objectWidth) +'/mnist'
    input_data_ = np.load(name + "_SDRs_" + data_set_section + ".npy")
    labels_ = np.load(name + "_labels_" + data_set_section + ".npy")

    idx = np.random.permutation(len(labels_))
    input_data = input_data_[idx]
    labels = labels_[idx]

    input_data_samples = []
    label_samples = []
    training_image_samples = []
    for mnist_iter in range(num_classes):
        indices = np.nonzero(labels == mnist_iter)

        input_data_samples.extend(input_data[indices][0: num_samples_per_class])
        label_samples.extend(labels[indices][0: num_samples_per_class])
        #training_image_samples.extend(images[indices][0: num_samples_per_class])
        
        assert (len(labels[indices][0: num_samples_per_class])
                == num_samples_per_class), "Insufficient training examples for loading"

    img_sdr_dic = {}
    feature_name = 0
    img_patch_list = []
    sample_counter = {str(i): 0 for i in range(num_classes)}
    for sample_iter in range(len(label_samples)):
        sample_temp = np.reshape(input_data_samples[sample_iter], (128, objectWidth, objectWidth))
        patch_loc_list = []

        for width_iter in range(objectWidth):
            for height_iter in range(objectWidth):
                feature_temp = sample_temp[:, width_iter, height_iter]
                indices = np.array(np.nonzero(feature_temp)[0], dtype="uint32")

                img_sdr_dic[str(feature_name)] = indices  # Name each feature uniquely

                patch_loc_list.append({"width": locationModuleWidth,
                                             "top": locationModuleWidth * width_iter,
                                             "height": locationModuleWidth,
                                             "name": str(feature_name),
                                             "left": locationModuleWidth * height_iter
                                             })

                feature_name += 1

        img_patch_list.append({"features": patch_loc_list,
                             "name": str(label_samples[sample_iter]) + "_"
                             + str(sample_counter[str(label_samples[sample_iter])])})

        sample_counter[str(label_samples[sample_iter])] += 1

    return img_sdr_dic, img_patch_list, training_image_samples


if __name__ == "__main__":

    generate_image_objects(
        data_set="mnist", numObjects=10,
        objectWidth=5, locationModuleWidth=50)
