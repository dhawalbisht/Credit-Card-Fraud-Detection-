import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras import regularizers
from scipy.stats import ks_2samp


# load data
data = pd.read_csv("creditcard.csv")
X = data.drop("Class", axis=1).values
y = data["Class"].values

# plot the class distribution
pd.value_counts(data["Class"])

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

bins = 100

# Plotting for Fraud transactions
ax1.hist(data.Time[data.Class == 1], bins=bins, color="red", alpha=0.7)
ax1.set_title("Fraud", fontsize=14)
ax1.set_ylabel("Number of Transactions", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.5)

# Plotting for Normal transactions
ax2.hist(data.Time[data.Class == 0], bins=bins, color="blue", alpha=0.7)
ax2.set_title("Normal", fontsize=14)
ax2.set_xlabel("Time (in Seconds)", fontsize=12)
ax2.set_ylabel("Number of Transactions", fontsize=12)
ax2.grid(True, linestyle="--", alpha=0.5)

# Remove top and right spines
for ax in [ax1, ax2]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(
    [data[data["Class"] == 1]["Amount"], data[data["Class"] == 0]["Amount"]],
    labels=["Fraud", "Normal"],
)

plt.title("Transaction Amounts: Fraud vs Normal")
plt.ylabel("Amount")
plt.yscale("log")
plt.show()

# split data into train, validation, and test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential(
    [
        # Adjusted number of neurons
        tf.keras.layers.Dense(
            128,
            activation="relu",
            input_shape=(X.shape[-1],),
            kernel_regularizer=regularizers.l2(0.001),
        ),  # L2 regularization
        tf.keras.layers.Dropout(0.2),  # Adjusted dropout rate
        tf.keras.layers.Dense(
            64, activation="relu", kernel_regularizer=regularizers.l2(0.001)
        ),  # L2 regularization
        tf.keras.layers.Dropout(0.2),  # Adjusted dropout rate
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()

metrics = [
    tf.keras.metrics.FalseNegatives(name="fn"),
    tf.keras.metrics.FalsePositives(name="fp"),
    tf.keras.metrics.TrueNegatives(name="tn"),
    tf.keras.metrics.TruePositives(name="tp"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]

# compile the model
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=metrics,
)

# configure early stopping
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=5
)

# calculate class weights
neg, pos = np.bincount(y_train)
total = neg + pos
class_weight = {0: 1, 1: 5}

# train the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=3,
    callbacks=[es],
    class_weight=class_weight,
)

# predict test data
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# score precision, recall, and f1
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Plot only the losses from history
losses = history.history["loss"]
val_losses = history.history["val_loss"]

plt.figure(figsize=(10, 7))
plt.plot(losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
