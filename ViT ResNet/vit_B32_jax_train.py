model_name = "imagenet21k_ViT-B_32"  # @param ["ViT-B_32", "Mixer-B_16"]
model_size = "ViT-B_32"
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
validation_size = 649
# dataset = "cifar10"
batch_size = 32
# batch_size = 16
config = common_config.with_dataset(common_config.get_config(), dataset)
config.batch = batch_size
config.pp.crop = 224  ## actually resize

pprint.pprint(config)
print("ds train processing...")
ds_train = input_pipeline.get_data_from_csv(
    config=config, csv="train_data.csv", mode="train"
)
ds_val = input_pipeline.get_data_from_csv(
    config=config, csv="validation_data.csv", mode="validation"
)
ds_test = input_pipeline.get_data_from_csv(
    config=config, csv="test_data.csv", mode="test"
)
num_classes = 2
del config

batch = next(iter(ds_train.as_numpy_iterator()))
images, labels, skin = batch["image"], batch["label"], batch["skin"]
print(f"image batch shape: {images.shape}")
print(f"image batch shape: {labels.shape}")
print(f"label batch shape: {skin.shape}")
print("image max", np.max(images))
print("image min", np.min(images))

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

print("variable:")
# pprint.pprint(variables)
print("variable params:")
for key, value in variables["params"].items():
    print(key)

# Load and convert pretrained checkpoint.
# This involves loading the actual pre-trained model results, but then also also
# modifying the parameters a bit, e.g. changing the final layers, and resizing
# the positional embeddings.
# For details, refer to the code and to the methods of the paper.
params = checkpoint.load_pretrained(
    pretrained_path=f"{model_name}.npz",
    init_params=variables["params"],
    model_config=model_config,
)


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


def get_accuracy(params_repl, output_file):
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

    for batch in ds_test.as_numpy_iterator():
        predicted = vit_apply_repl(params_repl, batch["image"])
        predicted = predicted.argmax(axis=-1).squeeze()
        label = batch["label"].argmax(axis=-1).squeeze()
        print(f"label shape = {label.shape}")
        print(f"predicted shape = {predicted.shape}")
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
            f"skin tone {skin}, correct = {metrics['correct']}, total = {metrics['total']}, accuracy = {metrics['correct']/metrics['total']}"
        )

    print("Binary skin groups")
    for skin, metrics in metrics_per_skin_tone2.items():
        print(
            f"skin tone {skin}, correct = {metrics['correct']}, total = {metrics['total']}, accuracy = {metrics['correct']/metrics['total']}",
        )

    with open(output_file, "w") as file:
        file.write("Test output:\n")
        file.write(f"\ntrain loss = {losses}")
        file.write(f"\nvalidation loss = {val_losses}")
        file.write(f"\nmodel learning rate = {lrs}")
        file.write(f"\nbatch_size = {batch_size}")
        file.write(f"\nepochs = {true_steps}")
        file.write(f"\nmax_stop_count = {max_stop_count}")
        file.write(f"\ngrad_norm_clip = {grad_norm_clip}")
        file.write(f"\noptimizer = {optimizer_type}")
        file.write(f"\naccum_steps = {accum_steps}")
        file.write(f"\nlearning_rate = {base_lr}")
        if fixed_lr:
            file.write(f"\nscheduler = None")
        else:
            file.write(f"\nwarmup_steps = {warmup_steps}")
            file.write(f"\nscheduler = CosineAnnealingLR")

        for skin, metrics in metrics_per_skin_tone.items():
            file.write(
                f"\nskin tone {skin} true label:{metrics['true']}\nskin tone {skin} predicted label:{metrics['predicted']}",
            )

        for skin, metrics in metrics_per_skin_tone2.items():
            file.write(
                f"\nbinary skin tone {skin} true label:{metrics['true']}\nbinary skin tone {skin} predicted label:{metrics['predicted']}",
            )
        file.write("\n\n")

    return good / total


"""### Fine-tune"""
# # 100 Steps take approximately 15 minutes in the TPU runtime.
total_steps = 1000
warmup_steps = 5
decay_type = "cosine"
# decay_type = "linear"
grad_norm_clip = 1
# This controls in how many forward passes the batch is split. 8 works well with
# a TPU runtime that has 8 devices. 64 should work on a GPU. You can of course
# also adjust the batch_size above, but that would require you to adjust the
# learning rate accordingly.
accum_steps = 4
# accum_steps = 2
base_lr = 0.001
fixed_lr = True
# fixed_lr = False

optimizer_type = "Adam"
# optimizer_type = "SGD"

# lr_fn = utils.create_learning_rate_schedule(
#     total_steps, base_lr, decay_type, warmup_steps
# )

if optimizer_type == "Adam":
    tx = optax.chain(
        optax.clip_by_global_norm(grad_norm_clip),
        optax.adam(learning_rate=base_lr),
        # optax.adam(learning_rate=lr_fn),
    )
elif optimizer_type == "SGD":
    tx = optax.chain(
        optax.clip_by_global_norm(grad_norm_clip),
        optax.sgd(
            # learning_rate=lr_fn,  # add lr scheduler function
            learning_rate=base_lr,  # add fixed lr
            momentum=0.9,
            accumulator_dtype="bfloat16",
        ),
    )

update_fn_repl = train.make_update_fn(
    apply_fn=model.apply, accum_steps=accum_steps, tx=tx
)
opt_state = tx.init(params)
opt_state_repl = flax.jax_utils.replicate(opt_state)

# Initialize PRNGs for dropout.
update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(0))


def val_loss_calc(
    params_repl,
):
    val_losses = []
    val_update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(1))
    val_update_fn_repl = train.make_val_update_fn(apply_fn=model.apply, accum_steps=1)

    steps = int(np.ceil(validation_size / batch_size))

    for _, val_batch in zip(range(steps), ds_val.as_numpy_iterator()):
        _, val_loss_repl, val_update_rng_repl = val_update_fn_repl(
            params_repl, val_batch, val_update_rng_repl
        )
        val_loss = float(val_loss_repl[0])
        val_losses.append(val_loss)
    loss = np.mean(val_losses)
    print("val loss = ", loss)
    return loss


losses = []
val_losses = []
lrs = []
# Completes in ~20 min on the TPU runtime.

min_val_loss = float("inf")
stop_count = 0
max_stop_count = 5
true_steps = 0
best_params_repl = None

for step, batch in zip(
    tqdm.trange(1, total_steps + 1),
    ds_train.as_numpy_iterator(),
):
    true_steps += 1
    params_repl, opt_state_repl, loss_repl, update_rng_repl = update_fn_repl(
        params_repl, opt_state_repl, batch, update_rng_repl
    )
    losses.append(float(loss_repl[0]))
    # lrs.append(float(lr_fn(step)))

    val_loss = val_loss_calc(params_repl)
    val_losses.append(float(val_loss))

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_params_single = flax.jax_utils.unreplicate(jax.device_get(params_repl))
        stop_count = 0
    else:
        stop_count += 1
        if stop_count == max_stop_count:
            print("Validation loss converges at step: ", true_steps)
            break

output_file = f"{model_name}_skin_{optimizer_type}_{'fixed' if fixed_lr else 'CosineAnnealingLR'}_{base_lr}_output.txt"
get_accuracy(flax.jax_utils.replicate(best_params_single), output_file)


ckpt_dir = os.path.abspath(os.getcwd())
os.makedirs(ckpt_dir, exist_ok=True)
custom_prefix = f"{model_name}_skin_{optimizer_type}_{'fixed' if fixed_lr else 'CosineAnnealingLR'}_{base_lr}_"

checkpoints.save_checkpoint(
    ckpt_dir=ckpt_dir,
    target=best_params_single,  # your trained parameters
    step=true_steps,
    prefix=custom_prefix,
    overwrite=True,  # overwrite if same step file exists
)
print(f"Checkpoint saved to {custom_prefix}")


"""### Inference"""

# # Download a pre-trained model.

# if model_name.startswith('Mixer'):
#   # Download model trained on imagenet2012
#   ![ -e "$model_name"_imagenet2012.npz ] || gsutil cp gs://mixer_models/imagenet1k/"$model_name".npz "$model_name"_imagenet2012.npz
#   model = models.MlpMixer(num_classes=1000, **model_config)
# else:
#   # Download model pre-trained on imagenet21k and fine-tuned on imagenet2012.
#   ![ -e "$model_name"_imagenet2012.npz ] || gsutil cp gs://vit_models/imagenet21k+imagenet2012/"$model_name".npz "$model_name"_imagenet2012.npz
#   model = models.VisionTransformer(num_classes=1000, **model_config)

# import os
# assert os.path.exists(f'{model_name}_imagenet2012.npz')

# # Load and convert pretrained checkpoint.
# params = checkpoint.load(f'{model_name}_imagenet2012.npz')
# params['pre_logits'] = {}  # Need to restore empty leaf for Flax.

# # Get imagenet labels.
# !wget https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt
# imagenet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))

# # Get a random picture with the correct dimensions.
# resolution = 224 if model_name.startswith('Mixer') else 384
# !wget https://picsum.photos/$resolution -O picsum.jpg
# import PIL
# img = PIL.Image.open('picsum.jpg')
# img

# # Predict on a batch with a single item (note very efficient TPU usage...)
# logits, = model.apply(dict(params=params), (np.array(img) / 128 - 1)[None, ...], train=False)

# preds = np.array(jax.nn.softmax(logits))
# for idx in preds.argsort()[:-11:-1]:
#   print(f'{preds[idx]:.5f} : {imagenet_labels[idx]}', end='')
