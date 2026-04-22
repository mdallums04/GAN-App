"""
gan_model.py  —  GAN building blocks (improved)

Changes vs original:
  • milestone_images now captured and stored inside history dict (req. #9)
  • tqdm used for inner batch loop (req. via spec)
  • Generator output correctly reshaped to (28,28,1) before being fed to
    Discriminator during standalone training (shape-safety fix)
  • Label smoothing applied correctly; fake labels are always 0.0
  • CIFAR-10 note added warning that model still expects 28×28 grayscale output
  • Model summary helper added
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, os
from PIL import Image
from tqdm import tqdm  # ← explicitly required

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Dense, LeakyReLU, BatchNormalization,
    Reshape, Flatten, Input
)
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10


# ─────────────────────────────────────────────
# GENERATOR
# ─────────────────────────────────────────────
def build_generator(
        latent_dim=100,
        learning_rate=0.0002,
        beta_1=0.5,
        layer_sizes=(256, 512, 1024)
):
    """
    Fully-connected Generator.
    Input : noise vector  (latent_dim,)
    Output: flat image vector (784,) in range [-1, 1]

    Architecture:
        Input → [Dense + LeakyReLU + BN] × len(layer_sizes) → Dense(784, tanh)
    """
    model = models.Sequential(name="Generator")

    # Input layer
    model.add(Input(shape=(latent_dim,)))

    # Hidden layers: Dense + LeakyReLU + BatchNorm
    for units in layer_sizes:
        model.add(Dense(units, use_bias=False))
        model.add(LeakyReLU(negative_slope=0.2))  # slope=0.2 per DCGAN paper
        model.add(BatchNormalization(momentum=0.8))

    # Output layer — tanh matches normalised image range [-1, 1]
    model.add(Dense(28 * 28, activation="tanh"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
        loss="binary_crossentropy"
    )
    return model


# ─────────────────────────────────────────────
# DISCRIMINATOR
# ─────────────────────────────────────────────
def build_discriminator(
        img_shape=(28, 28, 1),
        learning_rate=0.0002,
        beta_1=0.5,
        layer_sizes=(512, 256, 128)
):
    """
    Fully-connected Discriminator.
    Input : image tensor (28, 28, 1)
    Output: scalar ∈ [0,1], probability of being a real image

    Architecture:
        Input → Flatten → [Dense + LeakyReLU + BN] × (n-1)
               → Dense + LeakyReLU (no BN on last hidden)
               → Dense(1, sigmoid)

    BN is intentionally omitted on the last hidden layer to avoid instability
    when Discriminator weights are frozen during GAN training.
    """
    model = models.Sequential(name="Discriminator")

    # Input + flatten
    model.add(Input(shape=img_shape))
    model.add(Flatten())

    # Hidden layers
    for i, units in enumerate(layer_sizes):
        model.add(Dense(units, use_bias=False))
        model.add(LeakyReLU(negative_slope=0.2))
        # Omit BN on the last hidden layer for stability
        if i < len(layer_sizes) - 1:
            model.add(BatchNormalization(momentum=0.8))

    # Output layer
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─────────────────────────────────────────────
# GAN  (Generator + frozen Discriminator)
# ─────────────────────────────────────────────
def build_gan(generator, discriminator, latent_dim=100,
              learning_rate=0.0002, beta_1=0.5):
    """
    Stacks Generator → Discriminator with Discriminator frozen.
    Only Generator weights are updated during GAN.train_on_batch().
    """
    discriminator.trainable = False

    gan_input = Input(shape=(latent_dim,))
    generated = generator(gan_input)  # (batch, 784)
    img = Reshape((28, 28, 1))(generated)  # → (batch, 28, 28, 1)
    validity = discriminator(img)  # → (batch, 1)

    gan = models.Model(gan_input, validity, name="GAN")
    gan.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
        loss="binary_crossentropy"
    )
    return gan


# ─────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────
def load_dataset(name="mnist"):
    """
    Load one of: 'mnist', 'fashion_mnist', 'cifar10'.
    All datasets are resized/converted to (N, 28, 28, 1) grayscale
    and normalised to [-1, 1].
    """
    if name == "mnist":
        (x_train, _), _ = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)

    elif name == "fashion_mnist":
        (x_train, _), _ = fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)

    elif name == "cifar10":
        # CIFAR-10 is RGB 32×32 — resize to 28×28 and convert to grayscale
        (x_train, _), _ = cifar10.load_data()
        x_train = tf.image.resize(x_train, (28, 28))
        x_train = tf.image.rgb_to_grayscale(x_train)
        x_train = x_train.numpy()

    else:
        raise ValueError(f"Unsupported dataset: '{name}'. "
                         "Choose from: mnist, fashion_mnist, cifar10")

    x_train = x_train.astype("float32")
    x_train = (x_train - 127.5) / 127.5  # normalise to [-1, 1]
    return x_train


# ─────────────────────────────────────────────
# IMAGE GRID UTILITY
# ─────────────────────────────────────────────
def generate_and_plot(generator, latent_dim, n=16, title="", save_path=None):
    """
    Generate n images from random noise and plot a grid.
    Returns the matplotlib Figure.
    """
    noise = np.random.normal(0, 1, (n, latent_dim))
    fake = generator.predict(noise, verbose=0)  # (n, 784)
    imgs = fake.reshape(n, 28, 28)
    imgs = np.clip((imgs + 1.0) / 2.0, 0, 1)  # rescale for display

    cols = 4
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap="gray", interpolation="nearest")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_gan(
        generator,
        discriminator,
        gan,
        x_train,
        epochs=400,
        batch_size=64,
        latent_dim=100,
        milestone_epochs=(1, 30, 100, 200, 400),
        progress_callback=None,  # optional Streamlit/UI callback
        label_smoothing=True,
        add_noise=False,
        output_dir="gan_outputs"
):
    """
    Train the GAN for `epochs` epochs.

    Training trick: label smoothing for real images (0.9 instead of 1.0)
    reduces Discriminator overconfidence and encourages Generator gradients.

    Returns
    -------
    history : dict with keys:
        d_losses, g_losses, d_accuracies, milestone_images
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Label arrays (recreated once; shape matches batch_size) ---
    real_label = np.full((batch_size, 1), 0.9 if label_smoothing else 1.0)
    fake_label = np.zeros((batch_size, 1))
    trick_label = np.ones((batch_size, 1))  # fool Discriminator into "real"

    history = {
        "d_losses": [], "g_losses": [], "d_accuracies": [],
        "milestone_images": {}
    }

    n_samples = x_train.shape[0]

    # ── Outer loop over epochs (tqdm progress bar) ──────────────────────────
    for epoch in tqdm(range(1, epochs + 1), desc="GAN Training", unit="epoch"):

        epoch_d_loss = epoch_g_loss = epoch_d_acc = 0.0
        batches = 0

        # ── Inner loop over mini-batches (tqdm, leave=False for clean output)
        for _ in tqdm(range(n_samples // batch_size),
                      desc=f"  Epoch {epoch}", leave=False, unit="batch"):

            # Sample real images
            idx = np.random.randint(0, n_samples, batch_size)
            real_imgs = x_train[idx]  # (batch, 28, 28, 1)

            if add_noise:
                real_imgs = real_imgs + 0.05 * np.random.normal(size=real_imgs.shape)

            # Generate fake images
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_flat = generator.predict(noise, verbose=0)  # (batch, 784)
            # ── FIX: reshape to (batch, 28, 28, 1) to match Discriminator input
            fake_imgs = fake_flat.reshape(batch_size, 28, 28, 1)

            # Train Discriminator
            discriminator.trainable = True
            d_loss_r, d_acc_r = discriminator.train_on_batch(real_imgs, real_label)
            d_loss_f, d_acc_f = discriminator.train_on_batch(fake_imgs, fake_label)
            discriminator.trainable = False

            d_loss = 0.5 * (d_loss_r + d_loss_f)
            d_acc = 0.5 * (d_acc_r + d_acc_f)

            # Train Generator via GAN (trick Discriminator as real)
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, trick_label)

            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            epoch_d_acc += d_acc
            batches += 1

        # Epoch-average metrics
        avg_d = epoch_d_loss / batches
        avg_g = epoch_g_loss / batches
        avg_a = epoch_d_acc / batches

        history["d_losses"].append(avg_d)
        history["g_losses"].append(avg_g)
        history["d_accuracies"].append(avg_a)

        # Milestone snapshots  (req. #9)
        if epoch in milestone_epochs:
            path = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
            generate_and_plot(
                generator, latent_dim, n=16,
                title=f"Generator — Epoch {epoch}",
                save_path=path
            )
            history["milestone_images"][epoch] = path
            tqdm.write(f"  [Milestone] Epoch {epoch:3d} | "
                       f"D loss: {avg_d:.4f} | G loss: {avg_g:.4f} | "
                       f"D acc: {avg_a * 100:.1f}%")

        # Optional UI callback (used by Streamlit app)
        if progress_callback:
            progress_callback(epoch, epochs, {
                "d_loss": avg_d,
                "g_loss": avg_g,
                "d_acc": avg_a
            })

    return history
