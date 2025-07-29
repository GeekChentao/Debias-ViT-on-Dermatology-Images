model_name = "imagenet21k_R26+ViT-B_32"  # @param ["ViT-B_32", "Mixer-B_16"]
model_size = "R26+ViT-B_32"
import os

assert os.path.exists(f"{model_name}.npz")

from absl import logging
import flax
import jax
from flax.training import checkpoints

# from matplotlib import pyplot as plt
import numpy as np
import optax
import tqdm
import tensorflow as tf
import pandas as pd
import pprint

logging.set_verbosity(logging.INFO)

# Shows the number of available devices.
# In a CPU/GPU runtime this will be a single device.
# In a TPU runtime this will be 8 cores.
jax.local_devices()

import sys

if "./vision_transformer" not in sys.path:
    sys.path.append("./vision_transformer")

from vit_jax import checkpoint
from vit_jax import input_pipeline
from vit_jax import utils
from vit_jax import models
from vit_jax import train
from vit_jax.configs import common as common_config
from vit_jax.configs import models as models_config

# Helper functions for images.

labelnames = dict(
    skin=("benign", "malignant"),
)

dataset = "skin"
test_size = 648
# dataset = "cifar10"
batch_size = 512  # 16 32
config = common_config.with_dataset(common_config.get_config(), dataset)
config.batch = batch_size
config.pp.crop = 224  ## actually resize

ds_train = input_pipeline.get_data_from_csv(
    config=config, csv="train_data.csv", mode="train"
)
ds_val = input_pipeline.get_data_from_csv(
    config=config, csv="validation_data.csv", mode="test"
)
ds_test = input_pipeline.get_data_from_csv(
    config=config, csv="test_data.csv", mode="test"
)
num_classes = 2
del config

batch = next(iter(ds_train.as_numpy_iterator()))
images, labels, skin = batch["image"], batch["label"], batch["skin"]

"""### Load pre-trained"""
model_config = models_config.MODEL_CONFIGS[model_size]
print(f"\nmodel_config:")
pprint.pprint(model_config)

# Load model definition & initialize random parameters.
# This also compiles the model to XLA (takes some minutes the first time).
model = models.VisionTransformer(num_classes=num_classes, **model_config)

## just in time compile for faster processing
variables = jax.jit(
    lambda: model.init(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension of the batch for initialization.
        batch["image"][0, :1],
        train=False,
    ),
    backend="cpu",
)()


# Load and convert pretrained checkpoint.
# This involves loading the actual pre-trained model results, but then also also
# modifying the parameters a bit, e.g. changing the final layers, and resizing
# the positional embeddings.
# For details, refer to the code and to the methods of the paper.
# params = checkpoint.load_pretrained(
#     pretrained_path=f"{model_name}.npz",
#     init_params=variables["params"],
#     model_config=model_config,
# )

lr = 0.001
true_steps = 349
optimizer_type = "SGD"
# optimizer_type = "Adam"

scheduler = "fixed"
fixed_lr = True
# scheduler = "CosineAnnealingLR"
# fixed_lr = False

checkpoint_path = os.path.abspath(
    f"./{model_name}_skin_{optimizer_type}_{scheduler}_{lr}_{true_steps}"
)
output_file = f"{model_name}_skin_{optimizer_type}_{scheduler}_{lr}_output.txt"
params = checkpoints.restore_checkpoint(
    ckpt_dir=checkpoint_path, target=variables["params"]
)

print("Successfully load params")

"""### Evaluate"""

# So far, all our data is in the host memory. Let's now replicate the arrays
# into the devices.
# This will make every array in the pytree params become a ShardedDeviceArray
# that has the same data replicated across all local devices.
# For TPU it replicates the params in every core.
# For a single GPU this simply moves the data onto the device.
# For CPU it simply creates a copy.
params_repl = flax.jax_utils.replicate(params)
print(
    "params.cls:", type(params["head"]["bias"]).__name__, params["head"]["bias"].shape
)
print(
    "params_repl.cls:",
    type(params_repl["head"]["bias"]).__name__,
    params_repl["head"]["bias"].shape,
)

# Then map the call to our model's forward pass onto all available devices.
vit_apply_repl = jax.pmap(
    lambda params, inputs: model.apply(dict(params=params), inputs, train=False)
)


def get_accuracy(params_repl):
    """Returns accuracy evaluated on the test set."""
    metrics_per_skin_tone = {
        i: {"correct": 0, "total": 0, "true": list(), "predicted": list()}
        for i in range(1, 7)
    }
    metrics_per_skin_tone2 = {
        i: {"correct": 0, "total": 0, "true": list(), "predicted": list()}
        for i in range(1, 3)
    }

    good = total = 0
    steps = int(np.ceil(test_size / batch_size))

    for _, batch in zip(tqdm.trange(steps), ds_test.as_numpy_iterator()):
        predicted = vit_apply_repl(params_repl, batch["image"])
        predicted = predicted.argmax(axis=-1).squeeze()
        label = batch["label"].argmax(axis=-1).squeeze()
        is_same = predicted == label
        good += is_same.sum()
        total += len(is_same)
        skins = batch["skin"][0].flatten()
        for i, skin in enumerate(skins):
            metrics_per_skin_tone[skin]["correct"] += is_same[i]
            metrics_per_skin_tone[skin]["total"] += 1
            metrics_per_skin_tone[skin]["true"].append(int(label[i]))
            metrics_per_skin_tone[skin]["predicted"].append(int(predicted[i]))

            skin = 1 if skin in [1, 2, 3, 4] else 2
            metrics_per_skin_tone2[skin]["correct"] += is_same[i]
            metrics_per_skin_tone2[skin]["total"] += 1
            metrics_per_skin_tone2[skin]["true"].append(int(label[i]))
            metrics_per_skin_tone2[skin]["predicted"].append(int(predicted[i]))
    for skin, metrics in metrics_per_skin_tone.items():
        print(
            f"skin tone {skin}, correct = {metrics['correct']}, total = {metrics['total']}, accuracy = {metrics['correct']/metrics['total']},",
        )

    print("Binary skin groups")
    for skin, metrics in metrics_per_skin_tone2.items():
        print(
            f"skin tone {skin}, correct = {metrics['correct']}, total = {metrics['total']}, accuracy = {metrics['correct']/metrics['total']},",
        )

    with open(output_file, "w") as file:
        file.write("Test output:\n")
        # file.write(f"\ntrain loss = {losses}")
        # file.write(f"\nvalidation loss = {val_losses}")
        file.write(f"\nmodel learning rate = {lr}")
        file.write(f"\nbatch_size = {batch_size}")
        file.write(f"\nepochs = {true_steps}")
        # file.write(f"\nmax_stop_count = {max_stop_count}")
        file.write(f"\ngrad_norm_clip = {1.0}")
        file.write(f"\noptimizer = {optimizer_type}")
        file.write(f"\naccum_steps = {4}")
        file.write(f"\nlearning_rate = {lr}")
        file.write(f"\nscheduler = {scheduler}")

        for skin, metrics in metrics_per_skin_tone.items():
            file.write(
                f"\nskin tone {skin} true label:{metrics['true']}\nskin tone {skin} predicted label:{metrics['predicted']}",
            )

        for skin, metrics in metrics_per_skin_tone2.items():
            file.write(
                f"\nbinary skin tone {skin} true label:{metrics['true']}\nbinary skin tone {skin} predicted label:{metrics['predicted']}",
            )

    return good / total


accuracy = get_accuracy(params_repl)
