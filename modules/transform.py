import tensorflow as tf
import tensorflow_transform as tft
import re

# Fitur kategorikal dan numerik berdasarkan dataset
CATEGORICAL_FEATURES = {
    "Sex": 2,  # Male/Female
}

NUMERICAL_FEATURES = [
    "Red pixel",
    "Green pixel",
    "Blue pixel",
    "Hb",
]

# Create sanitized versions of feature names (remove spaces and special chars)
def sanitize_feature_name(name):
    """Sanitizes feature names to be compatible with TensorFlow."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

# Create sanitized feature mappings
FEATURE_MAP = {feature: sanitize_feature_name(feature) for feature in NUMERICAL_FEATURES}
FEATURE_MAP.update({feature: sanitize_feature_name(feature) for feature in CATEGORICAL_FEATURES})
FEATURE_MAP["Anaemic"] = "Anaemic"  # Keep label name the same

# Create sanitized feature lists
SANITIZED_NUMERICAL_FEATURES = [FEATURE_MAP[f] for f in NUMERICAL_FEATURES]
SANITIZED_CATEGORICAL_FEATURES = {FEATURE_MAP[f]: v for f, v in CATEGORICAL_FEATURES.items()}

LABEL_KEY = "Anaemic"  # Target label


def transformed_name(key):
    """
    Renames a feature key by first sanitizing it and then appending '_xf' to it.

    Args:
        key (str): The original feature key.

    Returns:
        str: The transformed feature key with '_xf' appended to it.
    """
    # First get the sanitized name from the map if it exists
    sanitized_key = FEATURE_MAP.get(key, key)
    return sanitized_key + '_xf'


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
    Convert a label (0 or 1) into a one-hot vector
    Args:
        int: label_tensor (0 or 1)
    Returns
        label tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs):
    """
    Preprocesses the input data by applying transformations to categorical and numerical features.

    Args:
        inputs (dict): A dictionary containing the input data. The keys are 
        the feature names and the values are the corresponding feature values.

    Returns:
        dict: A dictionary containing the preprocessed data. The keys are the transformed 
        feature names and the values are the transformed feature values.
    """
    outputs = {}

    for key, dim in CATEGORICAL_FEATURES.items():
        # Get sanitized name from map
        sanitized_key = FEATURE_MAP.get(key, key)
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1
        )
        outputs[transformed_name(sanitized_key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    for feature in NUMERICAL_FEATURES:
        # Get sanitized name from map
        sanitized_feature = FEATURE_MAP.get(feature, feature)
        outputs[transformed_name(sanitized_feature)] = tft.scale_to_0_1(inputs[feature])

    # Convert "Yes"/"No" string values to integers (1/0)
    sanitized_label = FEATURE_MAP.get(LABEL_KEY, LABEL_KEY)
    outputs[transformed_name(sanitized_label)] = tf.cast(
        tf.where(
            tf.equal(inputs[LABEL_KEY], "Yes"),
            tf.ones_like(inputs[LABEL_KEY], dtype=tf.int64),
            tf.zeros_like(inputs[LABEL_KEY], dtype=tf.int64)
        ),
        tf.int64
    )

    return outputs