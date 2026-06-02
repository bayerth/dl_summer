import argparse
import logging
import sys
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

SHAKESPEARE_URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
DEFAULT_BATCH_SIZE = 64


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def load_shakespeare_text():
    logger.info("Downloading/loading Shakespeare dataset")
    path_to_file = keras.utils.get_file("shakespeare.txt", SHAKESPEARE_URL)
    logger.debug("Dataset file: %s", path_to_file)
    with open(path_to_file) as file:
        text = file.read()
    logger.debug("Text length: %d characters", len(text))
    return text


def prepare_tokenizer(shakespeare_text):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, lower=True)
    tokenizer.fit_on_texts(shakespeare_text.lower())
    return tokenizer


def text_to_dataset(encoded_text, n_tokens, n_steps=100, batch_size=DEFAULT_BATCH_SIZE, one_hot=True):
    dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
    dataset = dataset.window(n_steps + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window_ds: window_ds.batch(n_steps + 1))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda window: (window[:, :-1], window[:, 1:]))
    if one_hot:
        dataset = dataset.map(lambda X, y: (tf.one_hot(X, depth=n_tokens), y))
    return dataset.repeat().prefetch(tf.data.AUTOTUNE)


def build_model(rnn_layers, rnn_type, n_tokens, embedding_dim=None):
    rnn_classes = {
        "GRU": keras.layers.GRU,
        "LSTM": keras.layers.LSTM,
        "SimpleRNN": keras.layers.SimpleRNN,
    }
    if rnn_type not in rnn_classes:
        available_types = ", ".join(rnn_classes)
        raise ValueError(f"Unsupported RNN type {rnn_type!r}. Use one of: {available_types}.")

    if embedding_dim is None:
        layers = [keras.layers.Input(shape=(None, n_tokens))]
    else:
        layers = [
            keras.layers.Input(shape=(None,), dtype="int32"),
            keras.layers.Embedding(input_dim=n_tokens, output_dim=embedding_dim),
        ]

    for neurons in rnn_layers:
        layers.append(rnn_classes[rnn_type](neurons, return_sequences=True))
    layers.append(keras.layers.Dense(n_tokens, activation="softmax"))

    model = keras.Sequential(layers)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model


def generated_name(rnn_layers, rnn_type, n_steps, epochs):
    layer_part = "_".join(str(neurons) for neurons in rnn_layers)
    return f"{rnn_type}_{layer_part}_s{n_steps}_e{epochs}"


def log_training_config(
    rnn_layers,
    rnn_type,
    n_steps,
    epochs,
    batch_size,
    embedding_dim,
    n_tokens,
    dataset_size,
    train_size,
    steps_per_epoch,
    directory,
    name,
):
    input_mode = "one-hot" if embedding_dim is None else f"embedding (dim={embedding_dim})"
    auto_name = generated_name(rnn_layers, rnn_type, n_steps, epochs)
    output_name = name or auto_name
    if not output_name.endswith(".keras"):
        output_name = f"{output_name}.keras"

    logger.info("=== Shakespeare RNN training configuration ===")
    logger.info(
        "Model name pattern: {type}_{layers}_s{n_steps}_e{epochs} "
        "(s=sequence length, e=max epochs)",
    )
    if name is None:
        logger.info("Auto-generated model name: %s", f"{auto_name}.keras")
    else:
        logger.info("Custom model name (--name): %s", output_name)
    logger.info("RNN type: %s", rnn_type)
    logger.info("RNN layers (neurons): %s (%d layer(s))", rnn_layers, len(rnn_layers))
    logger.info("Sequence length (n_steps): %d", n_steps)
    logger.info("Epochs (max): %d", epochs)
    logger.info("Batch size: %d", batch_size)
    logger.info("Input encoding: %s", input_mode)
    logger.info("Vocabulary size (n_tokens): %d", n_tokens)
    logger.info("Dataset size (characters): %d", dataset_size)
    logger.info("Train size (90%%): %d", train_size)
    logger.info("Steps per epoch: %d", steps_per_epoch)
    logger.info("Output directory: %s", directory)
    logger.info("Output model name: %s", output_name)
    logger.debug("TensorFlow version: %s", tf.__version__)
    logger.debug("Keras version: %s", keras.__version__)
    logger.debug(
        "Available devices: %s",
        [d.name for d in tf.config.list_physical_devices()],
    )


def save_model(model, name, rnn_layers, rnn_type, n_steps, epochs, directory="models"):
    model_name = name or generated_name(rnn_layers, rnn_type, n_steps, epochs)
    if not model_name.endswith(".keras"):
        model_name = f"{model_name}.keras"

    model_path = Path(directory) / model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return model_path


def run(
    rnn_layers,
    type="GRU",
    n_steps=100,
    epochs=20,
    name=None,
    directory="models",
    embedding_dim=None,
    batch_size=DEFAULT_BATCH_SIZE,
):
    if embedding_dim is not None and embedding_dim <= 0:
        raise ValueError("embedding_dim must be a positive integer or None")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    auto_name = generated_name(rnn_layers, type, n_steps, epochs)
    logger.info("=== Shakespeare RNN training runner started ===")
    logger.info(
        "Run parameters: type=%s, layers=%s, s%d (n_steps), e%d (epochs), batch_size=%d",
        type,
        rnn_layers,
        n_steps,
        epochs,
        batch_size,
    )
    logger.info(
        "Model name pattern: {type}_{layers}_s{n_steps}_e{epochs} (s=sequence length, e=max epochs)",
    )
    if name is None:
        logger.info("Planned model file: %s/%s.keras", directory, auto_name)
    else:
        model_file = name if name.endswith(".keras") else f"{name}.keras"
        logger.info("Planned model file: %s/%s", directory, model_file)

    shakespeare_text = load_shakespeare_text()
    tokenizer = prepare_tokenizer(shakespeare_text)
    n_tokens = len(tokenizer.word_index)
    dataset_size = tokenizer.document_count
    train_size = dataset_size * 90 // 100
    steps_per_epoch = train_size // batch_size

    [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
    logger.debug("Encoded sequence length: %d", len(encoded))
    logger.debug("Token index range: %d .. %d", int(encoded.min()), int(encoded.max()))
    logger.debug("Sample encoded tokens (first 20): %s", encoded[:20].tolist())

    log_training_config(
        rnn_layers=rnn_layers,
        rnn_type=type,
        n_steps=n_steps,
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        n_tokens=n_tokens,
        dataset_size=dataset_size,
        train_size=train_size,
        steps_per_epoch=steps_per_epoch,
        directory=directory,
        name=name,
    )

    train_dataset = text_to_dataset(
        encoded,
        n_tokens,
        n_steps=n_steps,
        batch_size=batch_size,
        one_hot=embedding_dim is None,
    )
    for X, y in train_dataset.take(1):
        logger.debug("Sample batch X shape: %s, y shape: %s", X.shape, y.shape)

    model = build_model(rnn_layers, type, n_tokens, embedding_dim=embedding_dim)
    logger.info("Model built: %d layer(s), %s trainable parameters", len(model.layers), f"{model.count_params():,}")
    model.summary()

    logger.info("Starting training")
    early_stopping = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.01, patience=2)
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[early_stopping],
        verbose=1,
    )

    epochs_run = len(history.history.get("loss", []))
    final_train_loss = history.history["loss"][-1] if epochs_run else float("nan")
    logger.info("Training finished after %d epoch(s), final train loss: %.4f", epochs_run, final_train_loss)
    if epochs_run < epochs:
        logger.info("Early stopping triggered (max epochs was %d)", epochs)

    logger.info("Evaluating model (%d steps)", steps_per_epoch)
    eval_loss = model.evaluate(train_dataset, steps=steps_per_epoch, verbose=1)
    logger.info("Evaluation loss: %.4f", eval_loss)

    model_path = save_model(
        model, name, rnn_layers, type, n_steps, epochs, directory=directory
    )
    logger.info("Model saved to %s", model_path)
    return model, history, eval_loss, model_path


def parse_rnn_layers(value):
    try:
        layers = [int(layer.strip()) for layer in value.split(",") if layer.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rnn_layers must be comma-separated integers") from exc

    if not layers:
        raise argparse.ArgumentTypeError("rnn_layers must contain at least one layer size")
    if any(layer <= 0 for layer in layers):
        raise argparse.ArgumentTypeError("all rnn_layers values must be positive")
    return layers


def parse_positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Shakespeare character RNN.")
    parser.add_argument(
        "--rnn-layers",
        required=True,
        type=parse_rnn_layers,
        help="Comma-separated neuron counts, e.g. 16,32 for two RNN layers.",
    )
    parser.add_argument("--type", default="GRU", choices=["GRU", "LSTM", "SimpleRNN"])
    parser.add_argument("--n-steps", default=100, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--name", default=None)
    parser.add_argument("--directory", default="models")
    parser.add_argument("--embedding-dim", default=None, type=parse_positive_int)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=parse_positive_int)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (DEBUG enables additional diagnostic output).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(getattr(logging, args.log_level))
    run(
        rnn_layers=args.rnn_layers,
        type=args.type,
        n_steps=args.n_steps,
        epochs=args.epochs,
        name=args.name,
        directory=args.directory,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
    )
