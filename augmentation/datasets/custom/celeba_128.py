from types import SimpleNamespace
import tensorflow as tf
import augmentation.datasets.utils
from collections import defaultdict

CELEBA_BASE_VARIANTS = ['5_o_Clock_Shadow',
                        'Arched_Eyebrows',
                        'Attractive',
                        'Bags_Under_Eyes',
                        'Bald',
                        'Bangs',
                        'Big_Lips',
                        'Big_Nose',
                        'Black_Hair',
                        'Blond_Hair',
                        'Blurry',
                        'Brown_Hair',
                        'Bushy_Eyebrows',
                        'Chubby',
                        'Double_Chin',
                        'Eyeglasses',
                        'Goatee',
                        'Gray_Hair',
                        'Heavy_Makeup',
                        'High_Cheekbones',
                        'Male',
                        'Mouth_Slightly_Open',
                        'Mustache',
                        'Narrow_Eyes',
                        'No_Beard',
                        'Oval_Face',
                        'Pale_Skin',
                        'Pointy_Nose',
                        'Receding_Hairline',
                        'Rosy_Cheeks',
                        'Sideburns',
                        'Smiling',
                        'Straight_Hair',
                        'Wavy_Hair',
                        'Wearing_Earrings',
                        'Wearing_Hat',
                        'Wearing_Lipstick',
                        'Wearing_Necklace',
                        'Wearing_Necktie',
                        'Young']

# train_group_sizes = {'Blond_Hair':
#                          {'Male':
#                               {(0, 0): 4054, (0, 1): 66874, (1, 0): 22880, (1, 1): 1387}  # 71629
#                           }
#                      }
#
# train_group_original_sizes = {'Blond_Hair':
#                                   {'Male':
#                                        {(0, 0): 71629, (0, 1): 66874, (1, 0): 22880, (1, 1): 1387}
#                                    }
#                               }
#
# val_group_sizes = {'Blond_Hair':
#                        {'Male':
#                             {(0, 0): 8535, (0, 1): 8276, (1, 0): 2874, (1, 1): 182}
#                         }
#                    }
#
# test_group_sizes = {'Blond_Hair':
#                         {'Male':
#                              {(0, 0): 9767, (0, 1): 7535, (1, 0): 2480, (1, 1): 180}
#                          }
#                     }

# GROUP_ORIGINAL_SIZES = {"train": train_group_original_sizes,
#                         "val": val_group_sizes}

train_group_sizes = defaultdict(dict)

train_group_original_sizes = defaultdict(dict)

val_group_sizes = defaultdict(dict)

test_group_sizes = defaultdict(dict)

GROUP_SIZE_DICTS = {"train_original": train_group_original_sizes,
                    "train": train_group_sizes,
                    "val": val_group_sizes,
                    "test": test_group_sizes}

SAVE_TFREC_NAME = None
LABEL_TYPE = None


# THIS IS TERRIBLE!!! ACCESSING THE LEN BY A DICTIONARY, NOT ACTUALLY CHECKING THE INPUT DATASET
def get_celeba_dataset_len(y_variant, z_variant, y_label, z_label):
    if y_label == -1:
        if z_label == -1:
            entries_to_sum = [(0, 0), (0, 1), (1, 0), (1, 1)]
        else:
            entries_to_sum = [(0, z_label), (1, z_label)]
    else:
        if z_label == -1:
            entries_to_sum = [(y_label, 0), (y_label, 1)]
        else:
            entries_to_sum = [(y_label, z_label)]

    return sum([train_group_sizes[y_variant][z_variant][k] for k in entries_to_sum]), \
           sum([val_group_sizes[y_variant][z_variant][k] for k in entries_to_sum]), \
           sum([test_group_sizes[y_variant][z_variant][k] for k in entries_to_sum])


def compute_celeba_dataset_len_single(y_variant, z_variant, y_label, z_label, dataset, dataset_name):
    """Compute subgroup sample size for a single set"""
    if y_label == -1:
        if z_label == -1:
            entries_to_populate = [(0, 0), (0, 1), (1, 0), (1, 1)]
        else:
            entries_to_populate = [(0, z_label), (1, z_label)]
    else:
        if z_label == -1:
            entries_to_populate = [(y_label, 0), (y_label, 1)]
        else:
            entries_to_populate = [(y_label, z_label)]

    def count_util(data):
        """Count entries in a dataset"""
        return sum([1 for _ in data])

    # Init with right keys
    global GROUP_SIZE_DICTS
    if len(GROUP_SIZE_DICTS[dataset_name].keys()) == 0:
        GROUP_SIZE_DICTS[dataset_name][y_variant][z_variant] = {(y_t, z_t): None for y_t, z_t in entries_to_populate}

    for y_t, z_t in entries_to_populate:
        dataset_subgroup = dataset.filter(lambda image, y, z: (y == y_t and z == z_t))
        # global GROUP_SIZE_DICTS
        GROUP_SIZE_DICTS[dataset_name][y_variant][z_variant][(y_t, z_t)] = count_util(dataset_subgroup)


def compute_celeba_dataset_len(y_variant, z_variant, y_label, z_label, train_data, val_data, test_data):
    """Compute subgroup sample size for the three sets: train, validation and test"""
    compute_celeba_dataset_len_single(y_variant, z_variant, y_label, z_label, train_data, "train")
    compute_celeba_dataset_len_single(y_variant, z_variant, y_label, z_label, val_data, "val")
    compute_celeba_dataset_len_single(y_variant, z_variant, y_label, z_label, test_data, "test")


def read_celeba_tfrecord(example, batched=True, parallelism=8):
    features = {"image": tf.io.FixedLenFeature([], tf.string)}
    features.update({CELEBA_BASE_VARIANTS[i]: tf.io.FixedLenFeature([], tf.int64)
                     for i in range(len(CELEBA_BASE_VARIANTS))})
    if batched:
        # Parse the TFRecord
        example = tf.io.parse_example(example, features)
        # Decode the image
        image = tf.map_fn(augmentation.datasets.utils.decode_raw_image, example['image'],
                          dtype=tf.uint8, back_prop=False, parallel_iterations=parallelism)
    else:
        # Parse the TFRecord
        example = tf.io.parse_single_example(example, features)
        # Decode the image
        image = augmentation.datasets.utils.decode_raw_image(example['image'])

    # Get all the other tags
    tags = {tag: example[tag] for tag in CELEBA_BASE_VARIANTS}

    return image, tags


def get_label_selection_function(label_type):
    if label_type == 'y':
        # Keep only the y labels
        return lambda image, y_label, z_label: (image, y_label)
    elif label_type == 'z':
        # Keep only the z labels
        return lambda image, y_label, z_label: (image, z_label)
    # CUSTOMISED ADD - TO USE WHEN SAVING THE DATA (for external usage)
    elif label_type == 'full':
        # Keep both x the z labels
        return lambda image, y_label, z_label: (image, y_label, z_label)
    else:
        raise NotImplementedError


def load_celeba_128(dataset_name, dataset_version, data_dir, save_tfrec_name, undersample_shuffle_seed):
    assert dataset_name.startswith('celeb_a_128'), \
        f'Dataset name is {dataset_name}, ' \
        f'should be celeb_a_128/<y_task>/<z_task>/<z_frac>/<which_y>/<which_z>/<label_type>/<optional_take_from_Y0Z0>'

    # Grab the names of the variants, the fraction of labeled z's available and the label type
    split_name = dataset_name.split("/")[1:]
    if len(split_name) == 6:
        y_variant, z_variant, z_frac, y_label, z_label, label_type = split_name
        n_y0z0_examples = -1
    elif len(split_name) == 7:
        y_variant, z_variant, z_frac, y_label, z_label, label_type, n_y0z0_examples = split_name
        n_y0z0_examples = int(n_y0z0_examples)
    else:
        raise NotImplementedError
    z_frac, y_label, z_label = float(z_frac), int(y_label), int(z_label)
    assert y_variant in CELEBA_BASE_VARIANTS, f'Dataset variant {y_variant} is not available.'
    assert z_variant in CELEBA_BASE_VARIANTS, f'Dataset variant {z_variant} is not available.'
    assert 0 <= z_frac <= 1, f'z_frac should be in [0, 1], not {z_frac}.'
    assert y_label in [-1, 0, 1], f'y_label should be in {-1, 0, 1}, not {y_label}.'
    assert z_label in [-1, 0, 1], f'z_label should be in {-1, 0, 1}, not {z_label}.'
    assert label_type in ['y', 'z'], f'Label types must be either y or z, not {label_type}.'
    assert n_y0z0_examples > 0 or n_y0z0_examples == -1, f'Please pass in a number greater than 0 or pass in -1.'
    assert z_frac == 1.0, 'z_frac has not been set up yet.'

    # Load up the list of .tfrec files for the train/val/test sets
    train_dataset = tf.data.Dataset.list_files(f'{data_dir}/train/*.tfrec', shuffle=False)
    val_dataset = tf.data.Dataset.list_files(f'{data_dir}/val/*.tfrec', shuffle=False)
    test_dataset = tf.data.Dataset.list_files(f'{data_dir}/test/*.tfrec', shuffle=False)

    # Construct the TF Dataset from the list of .tfrec files
    train_dataset = augmentation.datasets.utils. \
        get_dataset_from_list_files_dataset(train_dataset, proc_batch=128,
                                            tfrecord_example_reader=read_celeba_tfrecord).unbatch()

    val_dataset = augmentation.datasets.utils. \
        get_dataset_from_list_files_dataset(val_dataset, proc_batch=128,
                                            tfrecord_example_reader=read_celeba_tfrecord).unbatch()

    test_dataset = augmentation.datasets.utils. \
        get_dataset_from_list_files_dataset(test_dataset, proc_batch=128,
                                            tfrecord_example_reader=read_celeba_tfrecord).unbatch()

    # Map to grab the y and z labels for the attributes picked
    selection_fn = lambda image, tags: (image, int(tags[y_variant]), int(tags[z_variant]))
    train_dataset = train_dataset.map(selection_fn, num_parallel_calls=16)
    val_dataset = val_dataset.map(selection_fn, num_parallel_calls=16)
    test_dataset = test_dataset.map(selection_fn, num_parallel_calls=16)

    if y_label == 0 or y_label == 1:
        # Keep only one of the y_labels
        train_dataset = train_dataset.filter(lambda image, y, z: (y == y_label))
        val_dataset = val_dataset.filter(lambda image, y, z: (y == y_label))
        test_dataset = test_dataset.filter(lambda image, y, z: (y == y_label))

    if z_label == 0 or z_label == 1:
        # Keep only one of the z_labels
        train_dataset = train_dataset.filter(lambda image, y, z: (z == z_label))
        val_dataset = val_dataset.filter(lambda image, y, z: (z == z_label))
        test_dataset = test_dataset.filter(lambda image, y, z: (z == z_label))

    # Compute the sample size before undersampling the dataset
    compute_celeba_dataset_len_single(y_variant, z_variant, y_label, z_label, train_dataset, "train_original")

    import pdb;pdb.set_trace()
    # Filter out the Y0Z0 examples and then add a subset of them back in
    # here len(train_dataset) = 71629 (when loading the first of the 4 subgroups)
    if n_y0z0_examples > 0:
        # Take out examples from Y = 0, Z = 0
        # here len(train_dataset) = 4054 (when loading the first of the 4 subgroups)
        # Take the FIRST 4054 examples - BUT, WHEN IS IT SHUFFLING THE TRAINING SET?
        y_t, z_t = 0, 0  # TODO: Change it to be not hard coding
        if undersample_shuffle_seed != -1:
            train_dataset_y0z0 = train_dataset.filter(lambda image, y, z: (y == y_t and z == z_t)).take(n_y0z0_examples)
        else:
            train_dataset_y0z0 = train_dataset.filter(lambda image, y, z: (y == y_t and z == z_t))
            # SHUFFLE
            shuffle_buffer = train_group_original_sizes[y_variant][z_variant][(y_t, z_t)]
            train_dataset_y0z0 = train_dataset_y0z0.shuffle(shuffle_buffer, seed=undersample_shuffle_seed).take(n_y0z0_examples)

        # Keep only examples from groups other than Y = 0, Z = 0
        train_dataset = train_dataset.filter(lambda image, y, z: (y != 0 or z != 0))
        # Add the subset of Y = 0, Z = 0 examples back into the train dataset
        train_dataset = train_dataset.concatenate(train_dataset_y0z0)

        if save_tfrec_name is not None:
            # Crate global variable SAVE_TFREC_NAME
            global SAVE_TFREC_NAME
            SAVE_TFREC_NAME = save_tfrec_name
            global LABEL_TYPE
            LABEL_TYPE = label_type

            train_dataset_tosave = train_dataset
            # Save undersampled train set:
            label_selection_fn_tosave = get_label_selection_function("full")
            # Still 4054
            train_dataset_tosave = train_dataset_tosave.map(label_selection_fn_tosave, num_parallel_calls=16)
            record_file = f"/srv/galene0/sr572/celeba_128/undersampled_4054/{SAVE_TFREC_NAME}.tfrec"

            # import pdb;pdb.set_trace()
            with tf.io.TFRecordWriter(record_file) as writer:
                for sample in train_dataset_tosave:
                    tf_sample = customised_celeba_undersampled_tosave(sample)
                    writer.write(tf_sample.SerializeToString())

    # FINAL LEN - Here len(train_dataset) = 4054 (when loading the first of the 4 subgroups)
    # N.B. important compute it here before losing the information about z (after label_selection_fn)
    compute_celeba_dataset_len(y_variant, z_variant, y_label, z_label, train_dataset, val_dataset, test_dataset)
    # Get the label selection function and apply it
    label_selection_fn = get_label_selection_function(label_type)
    # Still 4054
    train_dataset = train_dataset.map(label_selection_fn, num_parallel_calls=16)
    val_dataset = val_dataset.map(label_selection_fn, num_parallel_calls=16)
    test_dataset = test_dataset.map(label_selection_fn, num_parallel_calls=16)

    # Compute the length of the training dataset
    # Here ERROR (MOST LIKELY) train_dataset_len = 71629
    train_dataset_len, val_dataset_len, test_dataset_len = get_celeba_dataset_len(y_variant,
                                                                                  z_variant,
                                                                                  y_label,
                                                                                  z_label)

    # import pdb;pdb.set_trace()
    # Make a dataset info namespace to ensure downstream compatibility
    num_classes = 2
    classes = [f'Not {z_variant}', f'{z_variant}'] if label_type == 'z' else [f'Not {y_variant}', f'{y_variant}']
    dataset_info = SimpleNamespace(features={'label': SimpleNamespace(num_classes=num_classes),
                                             'image': SimpleNamespace(shape=(128, 128, 3))},
                                   splits={'train': SimpleNamespace(num_examples=train_dataset_len),
                                           'val': SimpleNamespace(num_examples=val_dataset_len),
                                           'test': SimpleNamespace(num_examples=test_dataset_len)},
                                   # These 14 classes are in the same order as the labels in the dataset
                                   classes=classes)

    # Return the data sets
    return SimpleNamespace(dataset_info=dataset_info,
                           train_dataset=train_dataset,
                           val_dataset=val_dataset,
                           test_dataset=test_dataset)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def customised_celeba_undersampled_tosave(train_sample, label="full"):
    if label == "full":
        # Create a dictionary with features that may be relevant.
        feature = {
            # 'image': _int64_feature(train_sample[0].numpy()),
            # 'image': tf.data.Dataset.from_tensor_slices(sample[0]),
            # 'image': _bytes_feature(tf.io.serialize_tensor(train_sample[0])),
            'image': _bytes_feature(tf.image.encode_jpeg(train_sample[0]).numpy()),
            'y': _int64_feature(train_sample[1].numpy()),
            'z': _int64_feature(train_sample[2].numpy()),
        }
    else:
        feature = {
            'image': _bytes_feature(tf.image.encode_jpeg(train_sample[0]).numpy()),
            label: _int64_feature(train_sample[1].numpy())
        }

    return tf.train.Example(features=tf.train.Features(feature=feature))
