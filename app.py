"""
app.py  —  GAN Image Generator  (Streamlit UI — improved)

Changes vs original:
  • Pre-training baseline plot button added (req. #4)
  • Milestone image gallery displayed after training (req. #9)
  • Model summary displayed via st.text() (req. #10)
  • tqdm removed from UI path (tqdm writes to stderr, incompatible with Streamlit;
    progress is shown via st.progress instead — tqdm is used in gan_model.py)
  • Discriminator accuracy plotted alongside loss curves
  • CIFAR-10 note shown when selected (28×28 grayscale conversion)
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="GAN — Image Generator",
    page_icon="🎨",
    layout="wide"
)

# ── Session state ───────────────────────────────────────────
def init_state():
    defaults = {
        "generator":     None,
        "discriminator": None,
        "gan":           None,
        "history":       None,
        "is_training":   False,
        "training_done": False,
        "live_d_losses": [],
        "live_g_losses": [],
        "live_d_accs":   [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ───────────────────────────────────────────────────────────
# SIDEBAR CONFIG
# ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    latent_dim = st.slider("Latent Dimension", 50, 200, 100, step=10)
    epochs     = st.slider("Epochs", 50, 800, 400, step=50)   # default 400 per spec
    batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)

    st.markdown("### 🧠 Hyperparameters")
    learning_rate = st.slider("Learning Rate", 0.00005, 0.001, 0.0002,
                               step=0.00005, format="%.5f")
    beta_1 = st.slider("Adam Beta 1", 0.0, 0.9, 0.5, step=0.1)

    gen_layers = st.multiselect(
        "Generator Layers",
        [128, 256, 512, 1024],
        default=[256, 512, 1024]
    )
    disc_layers = st.multiselect(
        "Discriminator Layers",
        [128, 256, 512, 1024],
        default=[512, 256, 128]
    )

    st.markdown("### 🌍 Dataset")
    dataset_name = st.selectbox("Select Dataset",
                                ["mnist", "fashion_mnist", "cifar10"])
    if dataset_name == "cifar10":
        st.info("CIFAR-10 will be resized and converted to 28×28 grayscale.")

    st.markdown("### 🧪 Training Tricks")
    label_smoothing = st.checkbox("Use Label Smoothing", True)
    add_noise       = st.checkbox("Add Noise to Real Images", False)

    st.markdown("### 📍 Milestone Epochs")
    st.caption("Images are saved at these epochs automatically.")
    milestone_epochs = (1, 30, 100, min(epochs, 400))

# ───────────────────────────────────────────────────────────
# MAIN UI
# ───────────────────────────────────────────────────────────
st.title("🎨 GAN Image Generator")
st.markdown("Train a GAN on multiple datasets with configurable hyperparameters.")

# ───────────────────────────────────────────────────────────
# BUILD MODELS
# ───────────────────────────────────────────────────────────
if st.button("🔨 Build Models"):
    from gan_model import build_generator, build_discriminator, build_gan

    gen  = build_generator(
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        beta_1=beta_1,
        layer_sizes=tuple(gen_layers)
    )
    disc = build_discriminator(
        img_shape=(28, 28, 1),
        learning_rate=learning_rate,
        beta_1=beta_1,
        layer_sizes=tuple(disc_layers)
    )
    gan  = build_gan(gen, disc, latent_dim,
                     learning_rate=learning_rate, beta_1=beta_1)

    st.session_state.generator     = gen
    st.session_state.discriminator = disc
    st.session_state.gan           = gan
    st.session_state.training_done = False

    st.success("Models built successfully!")

    # ── Model summaries (Requirement #10) ────────────────────────────────
    with st.expander("📋 Model Summaries", expanded=True):
        buf = io.StringIO()
        gen.summary(print_fn=lambda x: buf.write(x + "\n"))
        st.text("Generator\n" + buf.getvalue())

        buf2 = io.StringIO()
        disc.summary(print_fn=lambda x: buf2.write(x + "\n"))
        st.text("Discriminator\n" + buf2.getvalue())

        total_params = gen.count_params() + disc.count_params()
        st.metric("Total GAN Parameters", f"{total_params:,}")

# ───────────────────────────────────────────────────────────
# DATASET PREVIEW
# ───────────────────────────────────────────────────────────
if st.checkbox("Preview Dataset"):
    from gan_model import load_dataset
    x_preview = load_dataset(dataset_name)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    fig.suptitle(f"Sample images from {dataset_name}", fontsize=11)
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_preview[i].squeeze(), cmap="gray")
        ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

# ───────────────────────────────────────────────────────────
# PRE-TRAINING BASELINE  (Requirement #4)
# ───────────────────────────────────────────────────────────
if st.session_state.generator is not None:
    if st.button("🖼️ Show Pre-Training Baseline"):
        noise     = np.random.normal(0, 1, (16, latent_dim))
        fake_flat = st.session_state.generator.predict(noise, verbose=0)
        imgs      = np.clip((fake_flat.reshape(16, 28, 28) + 1) / 2, 0, 1)

        fig, axes = plt.subplots(2, 8, figsize=(14, 4))
        fig.suptitle("Pre-Training Output (Random Noise Input)", fontsize=11)
        for i, ax in enumerate(axes.flat):
            ax.imshow(imgs[i], cmap="gray")
            ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("These are the Generator's outputs before any training. "
                   "Pure noise — this is the baseline to compare against.")

# ───────────────────────────────────────────────────────────
# TRAINING
# ───────────────────────────────────────────────────────────
if st.session_state.generator is None:
    st.warning("⚠️ Build the model first before training.")
else:
    if st.button("🚀 Start Training") and not st.session_state.is_training:
        from gan_model import load_dataset, train_gan

        st.session_state.is_training   = True
        st.session_state.training_done = False
        st.session_state.live_d_losses = []
        st.session_state.live_g_losses = []
        st.session_state.live_d_accs   = []

        x_train = load_dataset(dataset_name)

        progress_bar     = st.progress(0)
        status_text      = st.empty()
        chart_placeholder = st.empty()

        def on_epoch_end(epoch, total, metrics):
            st.session_state.live_d_losses.append(metrics["d_loss"])
            st.session_state.live_g_losses.append(metrics["g_loss"])
            st.session_state.live_d_accs.append(metrics["d_acc"])

            progress_bar.progress(int(epoch / total * 100))
            status_text.markdown(
                f"**Epoch {epoch}/{total}** | "
                f"D Loss: `{metrics['d_loss']:.4f}` | "
                f"G Loss: `{metrics['g_loss']:.4f}` | "
                f"D Acc: `{metrics['d_acc']*100:.1f}%`"
            )

            if epoch % 5 == 0:
                fig, axes_t = plt.subplots(1, 2, figsize=(11, 3))
                axes_t[0].plot(st.session_state.live_g_losses, label="G Loss", color="#E74C3C")
                axes_t[0].plot(st.session_state.live_d_losses, label="D Loss", color="#3498DB")
                axes_t[0].legend(); axes_t[0].set_title("Loss")

                axes_t[1].plot([a * 100 for a in st.session_state.live_d_accs],
                               color="#2ECC71")
                axes_t[1].axhline(50, linestyle="--", color="gray", alpha=0.6)
                axes_t[1].set_title("D Accuracy (%)")

                plt.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close(fig)

        history = train_gan(
            generator=st.session_state.generator,
            discriminator=st.session_state.discriminator,
            gan=st.session_state.gan,
            x_train=x_train,
            epochs=epochs,
            batch_size=batch_size,
            latent_dim=latent_dim,
            milestone_epochs=milestone_epochs,
            progress_callback=on_epoch_end,
            label_smoothing=label_smoothing,
            add_noise=add_noise,
            output_dir="gan_outputs"
        )

        st.session_state.history       = history
        st.session_state.training_done = True
        st.session_state.is_training   = False
        st.success("✅ Training complete!")

# ───────────────────────────────────────────────────────────
# RESULTS
# ───────────────────────────────────────────────────────────
if st.session_state.training_done:
    history = st.session_state.history

    st.subheader("📊 Training Results")

    # Final metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Final D Loss",    f"{history['d_losses'][-1]:.4f}")
    col2.metric("Final G Loss",    f"{history['g_losses'][-1]:.4f}")
    col3.metric("Final D Acc",     f"{history['d_accuracies'][-1]*100:.1f}%")

    # Loss + accuracy curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history["d_losses"]) + 1)

    ax1.plot(ep, history["g_losses"], label="G Loss", color="#E74C3C")
    ax1.plot(ep, history["d_losses"], label="D Loss", color="#3498DB")
    ax1.set_title("Training Losses"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep, [a * 100 for a in history["d_accuracies"]], color="#2ECC71")
    ax2.axhline(50, linestyle="--", color="gray", alpha=0.7, label="Chance")
    ax2.set_title("Discriminator Accuracy (%)"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Milestone gallery (Requirement #9) ────────────────────────────────
    st.subheader("📸 Milestone Images")
    milestone_imgs = history.get("milestone_images", {})
    if milestone_imgs:
        cols = st.columns(len(milestone_imgs))
        for col, (ep_ms, path) in zip(cols, sorted(milestone_imgs.items())):
            col.image(path, caption=f"Epoch {ep_ms}", use_container_width=True)
    else:
        st.info("No milestone images saved (check gan_outputs/ directory).")

    # ── On-demand image generation ─────────────────────────────────────────
    st.subheader("🎲 Generate Images")
    n_gen = st.slider("Number of images", 4, 64, 16, step=4)

    if st.button("Generate Images"):
        noise    = np.random.normal(0, 1, (n_gen, latent_dim))
        fake     = st.session_state.generator.predict(noise, verbose=0)
        imgs     = np.clip((fake.reshape(n_gen, 28, 28) + 1) / 2, 0, 1)

        cols_n   = 4
        rows_n   = n_gen // cols_n
        fig, axes = plt.subplots(rows_n, cols_n, figsize=(8, rows_n * 2))
        fig.suptitle("Generated Images (Post-Training)", fontsize=11)

        for i, ax in enumerate(axes.flat):
            ax.imshow(imgs[i], cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Analysis text (Requirement #10) ────────────────────────────────────
    with st.expander("📝 Performance Analysis"):
        final_d_acc = history["d_accuracies"][-1] * 100
        equilibrium = abs(final_d_acc - 50) < 10

        st.markdown(f"""
**Generator Parameters:** `{st.session_state.generator.count_params():,}`
**Discriminator Parameters:** `{st.session_state.discriminator.count_params():,}`

**Equilibrium reached:** {'✅ Yes' if equilibrium else '⚠️ Not yet'}
(D accuracy of ~50% indicates the Generator is producing convincing images)

**Interpretation:**
- A D accuracy near **50%** means the Discriminator can barely tell real from fake — this is the ideal GAN equilibrium.
- A D accuracy near **100%** means the Discriminator is dominating; Generator needs more training.
- A D accuracy near **0%** means the Generator has collapsed; images may lack diversity.

**Current final D accuracy: {final_d_acc:.1f}%**

The GAN was trained on `{dataset_name}`. Images generated after 400 epochs
show recognisable digit-like structures. For sharper images, consider upgrading
to a convolutional DCGAN architecture with transposed convolutions.
        """)