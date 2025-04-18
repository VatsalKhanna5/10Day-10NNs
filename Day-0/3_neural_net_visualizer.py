# streamlit_neural_visualizer.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# --- General Settings ---
st.set_page_config(page_title="Neural Net Learner", layout="wide")
sns.set_style("whitegrid")

# --- Data Generation ---
@st.cache_data
def generate_data(n_samples=3000, seed=0):
    np.random.seed(seed)

    # 1. Create a fine grid of sunlight and water
    grid_size = int(np.sqrt(n_samples * 10))  # generate 10x more for stratification
    sunlight = np.linspace(0, 10, grid_size)
    water = np.linspace(0, 5, grid_size)
    sun_grid, water_grid = np.meshgrid(sunlight, water)

    sunlight_flat = sun_grid.flatten()
    water_flat = water_grid.flatten()

    # 2. Normalize inputs
    sunlight_norm = sunlight_flat / 10
    water_norm = water_flat / 5

    # 3. Apply formula to get growth
    growth = 0.5 + 0.8 * np.sin(np.pi * sunlight_norm) * np.sqrt(water_norm) - 0.3 * (sunlight_norm ** 2) * water_norm
    growth = np.clip(growth, 0, 1)

    # 4. Bin growth values and sample equally from each bin
    df_full = pd.DataFrame({
        "sunlight": sunlight_flat,
        "water": water_flat,
        "sunlight_norm": sunlight_norm,
        "water_norm": water_norm,
        "growth": growth
    })

    bins = np.linspace(0, 1, 11)  # 10 bins (0.0–0.1, ..., 0.9–1.0)
    df_full['growth_bin'] = pd.cut(df_full['growth'], bins)

    # Sample equally from each bin
    samples_per_bin = n_samples // 10
    balanced_dfs = []
    for bin_range in df_full['growth_bin'].unique():
        bin_df = df_full[df_full['growth_bin'] == bin_range]
        if len(bin_df) >= samples_per_bin:
            sampled = bin_df.sample(samples_per_bin, random_state=seed)
            balanced_dfs.append(sampled)

    df_final = pd.concat(balanced_dfs).sample(frac=1, random_state=seed).drop(columns=['growth_bin'])

    return df_final.reset_index(drop=True)



# --- Activation Functions ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(output):
    return output * (1 - output)

# --- Neural Network Class ---
class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, seed=42):
        np.random.seed(seed)
        self.lr = lr
        self.weights_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_output = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.bias_output = np.zeros((1, output_size))

    def activation(self, z, func="tanh"):
        if func == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif func == "relu":
            return np.maximum(0, z)
        elif func == "tanh":
            return np.tanh(z)
        else:
            return z  # linear

    def activation_derivative(self, output, func="tanh"):
        if func == "sigmoid":
            return output * (1 - output)
        elif func == "relu":
            return (output > 0).astype(float)
        elif func == "tanh":
            return 1 - output ** 2
        else:
            return 1  # linear

    def forward(self, X):
        self.z_hidden = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.a_hidden = self.activation(self.z_hidden, func="tanh")
        self.z_output = np.dot(self.a_hidden, self.weights_output) + self.bias_output
        self.output = self.activation(self.z_output, func="sigmoid")  # Nonlinear output
        return self.output

    def backward(self, X, y):
        m = len(y)
        output_error = (self.output - y)
        d_output = output_error * self.activation_derivative(self.output, func="sigmoid")

        dW_out = np.dot(self.a_hidden.T, d_output)
        db_out = np.sum(d_output, axis=0, keepdims=True)

        hidden_error = np.dot(d_output, self.weights_output.T)
        d_hidden = hidden_error * self.activation_derivative(self.a_hidden, func="tanh")

        dW_hidden = np.dot(X.T, d_hidden)
        db_hidden = np.sum(d_hidden, axis=0, keepdims=True)

        self.weights_output -= self.lr * dW_out / m
        self.bias_output -= self.lr * db_out / m
        self.weights_hidden -= self.lr * dW_hidden / m
        self.bias_hidden -= self.lr * db_hidden / m

    def train_one_epoch(self, X, y):
        self.forward(X)
        self.backward(X, y)
        loss = np.mean((self.output - y) ** 2)
        return loss

    def predict(self, X):
        return self.forward(X)


# --- Visualization Function ---
def draw_network(weights_hidden, weights_output, activations=None, ax=None):
    G = nx.DiGraph()
    pos = {}
    layer_nodes = {0: [], 1: [], 2: []}

    input_size = weights_hidden.shape[0]
    hidden_size = weights_hidden.shape[1]
    output_size = weights_output.shape[1]

    # Add nodes
    for i in range(input_size):
        node = f"I{i}"
        G.add_node(node)
        pos[node] = (0, -i)
        layer_nodes[0].append(node)

    for i in range(hidden_size):
        node = f"H{i}"
        G.add_node(node)
        pos[node] = (1, -i)
        layer_nodes[1].append(node)

    for i in range(output_size):
        node = f"O{i}"
        G.add_node(node)
        pos[node] = (2, -i)
        layer_nodes[2].append(node)

    # Add edges
    for i in range(input_size):
        for j in range(hidden_size):
            G.add_edge(f"I{i}", f"H{j}", weight=weights_hidden[i, j])

    for i in range(hidden_size):
        for j in range(output_size):
            G.add_edge(f"H{i}", f"O{j}", weight=weights_output[i, j])

    # Edge weights
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    norm = plt.Normalize(-1, 1)
    cmap = plt.cm.seismic

    nx.draw(G, pos, ax=ax, with_labels=True, edge_color=edge_colors, edge_cmap=cmap,
            node_color='lightblue', node_size=1500, width=2, arrows=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
    ax.set_title("Neural Network Architecture with Weights")

# --- App Interface ---
st.title("🌱 Neural Network Visual Learning Tool")
st.markdown("A fully interactive journey into training a basic neural network on simulated plant growth data.")

with st.sidebar:
    st.header("🧠 Network Config")
    hidden_size = st.slider("Hidden Layer Size", 2, 20, 5)
    learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
    epochs = st.slider("Epochs", 100, 5000, 1000, step=100)
    seed = st.number_input("Random Seed", value=42)

    if st.button("🔄 Initialize Network"):
        st.session_state.nn = SimpleNeuralNet(2, hidden_size, 1, lr=learning_rate, seed=seed)
        st.session_state.history = []
        st.session_state.epoch = 0
        st.session_state.trained = False

# --- Initialization ---
if "nn" not in st.session_state:
    st.session_state.nn = SimpleNeuralNet(2, 5, 1)
    st.session_state.history = []
    st.session_state.epoch = 0
    st.session_state.trained = False

# --- Data ---
data = generate_data()
X = data[["sunlight_norm", "water_norm"]].values
y = data["growth"].values.reshape(-1, 1)

# --- Training ---
st.subheader("📊 Train the Network")

col_train, col_vis = st.columns(2)

with col_train:
    if st.button("👣 Step Train (1 Epoch)"):
        loss = st.session_state.nn.train_one_epoch(X, y)
        st.session_state.history.append(loss)
        st.session_state.epoch += 1
        st.session_state.trained = True
        st.rerun()
    
    # Add this just below "if st.button("👣 Step Train (1 Epoch)")" section

    if st.button("🚀 Train Automatically"):
        with st.spinner("Training in progress..."):
            for _ in range(epochs):
                loss = st.session_state.nn.train_one_epoch(X, y)
                st.session_state.history.append(loss)
                st.session_state.epoch += 1
            st.session_state.trained = True
        st.success(f"Auto Training Complete! ✅ Trained for {epochs} more epochs.")
        st.rerun()


    if st.session_state.history:
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(st.session_state.history, label="MSE Loss", color="purple")
        ax_loss.set_title("Loss over Epochs")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        st.pyplot(fig_loss)

    if st.session_state.trained:
        preds = st.session_state.nn.predict(X)
        r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
        st.metric("Final R² Score", f"{r2:.4f}")
        st.metric("Current Epoch", st.session_state.epoch)

with col_vis:
    fig_net, ax_net = plt.subplots(figsize=(6, 6))
    draw_network(st.session_state.nn.weights_hidden, st.session_state.nn.weights_output, ax=ax_net)
    st.pyplot(fig_net)

# --- Prediction Interface ---
st.subheader("🔮 Make a Prediction")
sun = st.slider("Sunlight (hours)", 0.0, 10.0, 5.0)
water = st.slider("Water (liters)", 0.0, 5.0, 2.5)
input_val = np.array([[sun / 10, water / 5]])

if st.session_state.trained:
    pred = st.session_state.nn.predict(input_val)[0][0]
    st.success(f"🌿 Predicted Growth: **{pred:.4f}**")
else:
    st.warning("Train the network to see predictions.")

# --- Additional Visualizations ---
if st.session_state.trained:
    st.subheader("📈 Additional Graphs for Deeper Insight")

    pred_all = st.session_state.nn.predict(X)
    r2_scores = [1 - np.sum((y - st.session_state.nn.predict(X))**2) / np.sum((y - np.mean(y))**2)
                 for _ in st.session_state.history]

    col1, col2 = st.columns(2)

    # 1. Predicted vs Actual (for all training samples)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.scatter(y, pred_all, alpha=0.6, color="teal")
        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax1.set_xlabel("Actual Growth")
        ax1.set_ylabel("Predicted Growth")
        ax1.set_title("Predicted vs Actual (Training Set)")
        st.pyplot(fig1)

    # 2. Predicted Value vs Real Formula
    with col2:
        def true_formula(sun, water):
            s = sun / 10
            w = water / 5
            return np.clip(0.5 + 0.8 * np.sin(np.pi * s) * np.sqrt(w) - 0.3 * (s ** 2) * w, 0, 1)

        true_val = true_formula(sun, water)

        fig2, ax2 = plt.subplots()
        ax2.bar(["Model Prediction", "True (Formula)"], [pred, true_val], color=["orange", "green"])
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Growth Value")
        ax2.set_title("User Input Prediction vs True Value")
        st.pyplot(fig2)

    # 3. R² Accuracy Over Time
    st.markdown("### 📊 R² Score Progress")
    fig3, ax3 = plt.subplots()
    ax3.plot(r2_scores, color="darkred", linewidth=2)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("R² Score")
    ax3.set_title("Accuracy (R²) Over Training")
    ax3.set_ylim(0, 1)
    st.pyplot(fig3)


st.markdown("---")
st.markdown("Made with ❤️ for students learning about neural nets. Fine-tune, visualize, and play around to understand how training affects the network.")
