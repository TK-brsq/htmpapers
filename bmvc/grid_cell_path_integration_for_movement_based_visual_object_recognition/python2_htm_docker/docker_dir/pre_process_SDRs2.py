
"""
Load sparse distributed representations (SDRs) genereated from a k-Winner
Take All convolutional neural network, and constructs "objects"
compatible with the Columns Plus object-recognition algorithm
"""

import numpy as np


def generate_image_objects(
        data_set, num_samples_per_class, objectWidth,  # noqa: N803
        locationModuleWidth,  # noqa: N803
        data_set_section="SDR_classifiers_training", block_split=None):

    #print("Loading " + data_set_section + " data-set from " + data_set)
    input_data = np.load("training_and_testing_data2/" + data_set + "_SDRs_"
                         + data_set_section + ".npy")
    labels = np.load("training_and_testing_data2/" + data_set + "_labels_"
                     + data_set_section + ".npy")
    images = np.load("training_and_testing_data2/" + data_set + "_images_"
                     + data_set_section + ".npy")

    input_data_samples = []
    label_samples = []
    training_image_samples = []

    if block_split is not None:
        print("\nOnly training on the first " + str(block_split) + " classes")
        num_classes = block_split
    else:
        num_classes = 10

    for mnist_iter in range(num_classes):
        indices = np.nonzero(labels == mnist_iter)

        # Get num_samples_per_class of each digit/class type
        input_data_samples.extend(input_data[indices][0: num_samples_per_class])
        label_samples.extend(labels[indices][0: num_samples_per_class])
        training_image_samples.extend(images[indices][0: num_samples_per_class])

        assert (len(labels[indices][0: num_samples_per_class])
                == num_samples_per_class), "Insufficient training examples for loading"

    features_dic = {}
    feature_name = 0
    width_one = locationModuleWidth

    objects_list = []

    # Keep track of how many exampels of particular MNIST digits have come up; used
    # to name unique samples iteratively
    sample_counter = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0}

    for sample_iter in range(len(label_samples)):

        sample_temp = np.reshape(input_data_samples[sample_iter],
                                 (128, objectWidth, objectWidth))
        sample_features_list = []

        for width_iter in range(objectWidth):
            for height_iter in range(objectWidth):

                # Convert the SDRs into sparse arrays (i.e. just representing the
                # non-zero elements)
                feature_temp = sample_temp[:, width_iter, height_iter]
                indices = np.array(np.nonzero(feature_temp)[0], dtype="uint32")

                # The location of the feature as expected by the Columns Plus-style
                # object
                top = width_one * width_iter
                left = width_one * height_iter

                features_dic[str(feature_name)] = indices  # Name each feature uniquely

                sample_features_list.append({"width": width_one,
                                             "top": top,
                                             "height": width_one,
                                             "name": str(feature_name),
                                             "left": left
                                             })

                feature_name += 1

        objects_list.append({"features": sample_features_list,
                             "name": str(label_samples[sample_iter]) + "_"
                             + str(sample_counter[str(label_samples[sample_iter])])})

        sample_counter[str(label_samples[sample_iter])] += 1

    #print("Number of samples for each class ")
    #print(sample_counter)

    return features_dic, objects_list, training_image_samples


if __name__ == "__main__":

    generate_image_objects(
        data_set="mnist", numObjects=10,
        objectWidth=7, locationModuleWidth=50)
