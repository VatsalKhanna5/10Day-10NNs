import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# --- General Settings ---
st.set_page_config(page_title="Neural Net Learner", layout="wide")
sns.set_style("whitegrid")

# --- Realistic Plant Growth Model ---
def realistic_plant_growth(sunlight, water, temperature=25):
    """
    A more realistic plant growth model based on:
    - Sunlight follows a bell curve (plants need sufficient light but can get burned)
    - Water shows diminishing returns
    - Temperature has an optimal range
    - All factors interact with each other
    """
    # Normalize inputs to 0-1 range
    sun_norm = sunlight / 10
    water_norm = water / 5
    temp_norm = (temperature - 5) / 30  # Normalize to 0-1 for range 5-35°C
    
    # Sunlight response (bell curve with optimal around 6-7 hours)
    sun_effect = 4 * sun_norm * (1 - sun_norm) ** 0.5
    
    # Water response (sigmoid-like with diminishing returns)
    water_effect = 1 - np.exp(-3 * water_norm)
    
    # Temperature effect (bell curve centered around 23°C)
    temp_effect = np.exp(-((temp_norm - 0.6) ** 2) / 0.1)
    
    # Interaction effects
    sun_water_interaction = 0.2 * sun_norm * water_norm
    temp_water_interaction = -0.1 * temp_norm * (1 - water_norm)
    sun_temp_interaction = -0.2 * (sun_norm - 0.6) * (temp_norm - 0.6)
    
    # Final growth rate calculation
    growth_rate = 0.3 + (
        0.4 * sun_effect + 
        0.3 * water_effect + 
        0.2 * temp_effect +
        sun_water_interaction +
        temp_water_interaction +
        sun_temp_interaction
    )
    
    # Constrain to 0-1 range
    return np.clip(growth_rate, 0, 1)

# --- Data Generation ---
@st.cache_data
def generate_data(n_samples=3000, seed=0):
    np.random.seed(seed)
    
    # Create grid points for stratified sampling
    grid_size = int(np.sqrt(n_samples * 5))  # Generate more points for stratification
    sunlight = np.linspace(0, 10, grid_size)
    water = np.linspace(0, 5, grid_size)
    temperatures = np.linspace(5, 35, max(5, grid_size // 8))  # Fewer temperature points
    
    # Create combinations
    combinations = []
    for s in sunlight:
        for w in water:
            for t in np.random.choice(temperatures, 3):  # Select random temperatures to reduce combinations
                combinations.append([s, w, t])
    
    # Sample from combinations if too many
    if len(combinations) > n_samples * 5:
        sample_indices = np.random.choice(len(combinations), n_samples * 5, replace=False)
        combinations = [combinations[i] for i in sample_indices]
    
    # Create dataframe and calculate growth
    df_samples = pd.DataFrame(combinations, columns=['sunlight', 'water', 'temperature'])
    df_samples['growth'] = df_samples.apply(
        lambda row: realistic_plant_growth(row['sunlight'], row['water'], row['temperature']), 
        axis=1
    )
    
    # Normalize inputs
    df_samples['sunlight_norm'] = df_samples['sunlight'] / 10
    df_samples['water_norm'] = df_samples['water'] / 5
    df_samples['temp_norm'] = (df_samples['temperature'] - 5) / 30
    
    # Stratify samples
    bins = np.linspace(0, 1, 11)  # 10 bins (0.0–0.1, ..., 0.9–1.0)
    df_samples['growth_bin'] = pd.cut(df_samples['growth'], bins)
    
    # Sample equally from each bin
    balanced_dfs = []
    samples_per_bin = n_samples // 10
    
    for bin_range in df_samples['growth_bin'].unique():
        bin_df = df_samples[df_samples['growth_bin'] == bin_range]
        if len(bin_df) >= samples_per_bin:
            sampled = bin_df.sample(samples_per_bin, random_state=seed)
            balanced_dfs.append(sampled)
        else:
            balanced_dfs.append(bin_df)  # Take all if not enough samples
    
    df_final = pd.concat(balanced_dfs).sample(frac=1, random_state=seed).drop(columns=['growth_bin'])
    
    return df_final.reset_index(drop=True)

# --- Neural Network Class with Multiple Hidden Layers ---
class EnhancedNeuralNet:
    def __init__(self, input_size, hidden_layers, output_size, lr=0.01, seed=42):
        """
        Initialize a neural network with multiple hidden layers
        
        Parameters:
        - input_size: number of input features
        - hidden_layers: list of sizes for each hidden layer
        - output_size: number of output units
        - lr: learning rate
        - seed: random seed for weight initialization
        """
        np.random.seed(seed)
        self.lr = lr
        self.n_layers = len(hidden_layers) + 1  # hidden layers + output layer
        self.hidden_activations = ["tanh"] * len(hidden_layers)
        self.output_activation = "sigmoid"
        
        # Initialize weights and biases for all layers
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_layers[0]) * np.sqrt(2. / input_size))
        self.biases.append(np.zeros((1, hidden_layers[0])))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.weights.append(np.random.randn(hidden_layers[i], hidden_layers[i+1]) * 
                               np.sqrt(2. / hidden_layers[i]))
            self.biases.append(np.zeros((1, hidden_layers[i+1])))
        
        # Last hidden layer to output
        self.weights.append(np.random.randn(hidden_layers[-1], output_size) * 
                           np.sqrt(2. / hidden_layers[-1]))
        self.biases.append(np.zeros((1, output_size)))

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
        self.layer_inputs = []  # Store inputs to each layer (needed for backprop)
        self.layer_outputs = []  # Store outputs of each layer
        
        # Input layer (no activation)
        current_input = X
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            self.layer_inputs.append(current_input)
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            a = self.activation(z, func=self.hidden_activations[i])
            self.layer_outputs.append(a)
            current_input = a
        
        # Output layer
        self.layer_inputs.append(current_input)
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        output = self.activation(z_output, func=self.output_activation)
        self.layer_outputs.append(output)
        
        return output

    def backward(self, X, y):
        m = len(y)
        output = self.layer_outputs[-1]
        
        # Output layer error
        d_layer = (output - y) * self.activation_derivative(output, func=self.output_activation)
        deltas = [d_layer]
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 1, 0, -1):
            d_layer = np.dot(deltas[0], self.weights[i].T) * self.activation_derivative(
                self.layer_outputs[i-1], func=self.hidden_activations[i-1])
            deltas.insert(0, d_layer)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            layer_input = X if i == 0 else self.layer_outputs[i-1]
            self.weights[i] -= self.lr * np.dot(layer_input.T, deltas[i]) / m
            self.biases[i] -= self.lr * np.sum(deltas[i], axis=0, keepdims=True) / m

    def train_one_epoch(self, X, y):
        self.forward(X)
        self.backward(X, y)
        loss = np.mean((self.layer_outputs[-1] - y) ** 2)
        return loss

    def predict(self, X):
        return self.forward(X)

# --- Visualization Function ---
def draw_network(nn, ax=None):
    # Count the total number of nodes
    input_size = nn.weights[0].shape[0]
    hidden_layers = [w.shape[1] for w in nn.weights[:-1]]
    output_size = nn.weights[-1].shape[1]
    total_layers = len(hidden_layers) + 2  # +2 for input and output layers
    
    G = nx.DiGraph()
    pos = {}
    layer_nodes = {i: [] for i in range(total_layers)}
    
    # Add input nodes
    for i in range(input_size):
        node = f"I{i}"
        G.add_node(node)
        pos[node] = (0, -i * 2)
        layer_nodes[0].append(node)
    
    # Add hidden layer nodes
    for layer_idx, layer_size in enumerate(hidden_layers):
        for i in range(layer_size):
            node = f"H{layer_idx}_{i}"
            G.add_node(node)
            pos[node] = (layer_idx + 1, -i * 2)
            layer_nodes[layer_idx + 1].append(node)
    
    # Add output nodes
    for i in range(output_size):
        node = f"O{i}"
        G.add_node(node)
        pos[node] = (total_layers - 1, -i * 2)
        layer_nodes[total_layers - 1].append(node)
    
    # Add edges between input and first hidden layer
    for i in range(input_size):
        for j in range(hidden_layers[0]):
            G.add_edge(f"I{i}", f"H0_{j}", weight=nn.weights[0][i, j])
    
    # Add edges between hidden layers
    for layer_idx in range(len(hidden_layers) - 1):
        for i in range(hidden_layers[layer_idx]):
            for j in range(hidden_layers[layer_idx + 1]):
                G.add_edge(f"H{layer_idx}_{i}", f"H{layer_idx+1}_{j}", 
                           weight=nn.weights[layer_idx + 1][i, j])
    
    # Add edges between last hidden layer and output
    for i in range(hidden_layers[-1]):
        for j in range(output_size):
            G.add_edge(f"H{len(hidden_layers)-1}_{i}", f"O{j}", 
                       weight=nn.weights[-1][i, j])
    
    # Edge weights
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    norm = plt.Normalize(-1, 1)
    cmap = plt.cm.seismic
    
    # Draw network
    nx.draw(G, pos, ax=ax, with_labels=True, edge_color=edge_colors, edge_cmap=cmap,
            node_color='lightblue', node_size=1200, width=2, arrows=True,
            font_size=8)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
    ax.set_title("Neural Network Architecture with Weights")

# --- App Interface ---
st.title("🌱 Enhanced Neural Network Visual Learning Tool")
st.markdown("A fully interactive journey into training neural networks with customizable architectures on realistic plant growth data.")

with st.sidebar:
    st.header("🧠 Network Config")
    
    # Network architecture settings
    st.subheader("Architecture")
    num_hidden_layers = st.slider("Number of Hidden Layers", 1, 5, 1)
    
    # Create layer size inputs
    hidden_layers = []
    for i in range(num_hidden_layers):
        layer_size = st.slider(f"Neurons in Hidden Layer {i+1}", 2, 50, 5)
        hidden_layers.append(layer_size)
    
    # Input features selection
    st.subheader("Input Features")
    use_temperature = st.checkbox("Include Temperature as Input", True)
    input_size = 3 if use_temperature else 2
    
    # Training parameters
    st.subheader("Training Parameters")
    learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
    epochs = st.slider("Auto-Train Epochs", 100, 5000, 1000, step=100)
    seed = st.number_input("Random Seed", value=42)
    
    if st.button("🔄 Initialize Network"):
        st.session_state.nn = EnhancedNeuralNet(input_size, hidden_layers, 1, lr=learning_rate, seed=seed)
        st.session_state.history = []
        st.session_state.epoch = 0
        st.session_state.trained = False
        st.session_state.use_temperature = use_temperature
        st.success(f"Network initialized with {len(hidden_layers)} hidden layers!")

# --- Initialization ---
if "nn" not in st.session_state:
    # Default initialization with a single hidden layer
    st.session_state.nn = EnhancedNeuralNet(3, [5], 1)
    st.session_state.history = []
    st.session_state.epoch = 0
    st.session_state.trained = False
    st.session_state.use_temperature = True

# --- Data ---
data = generate_data()

# Select features based on user choice
if st.session_state.use_temperature:
    X = data[["sunlight_norm", "water_norm", "temp_norm"]].values
else:
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
    if len(st.session_state.nn.weights) <= 3:  # Only draw if network is not too complex
        fig_net, ax_net = plt.subplots(figsize=(8, 8))
        draw_network(st.session_state.nn, ax=ax_net)
        st.pyplot(fig_net)
    else:
        st.info("Network visualization is simplified for complex architectures with many layers.")
        # Create a simpler visualization for complex networks
        fig_net, ax_net = plt.subplots(figsize=(8, 4))
        
        # Display a simplified diagram
        layer_sizes = [st.session_state.nn.weights[0].shape[0]] + \
                      [w.shape[1] for w in st.session_state.nn.weights]
        
        ax_net.axis('off')
        max_neurons = max(layer_sizes)
        
        for i, size in enumerate(layer_sizes):
            for j in range(size):
                circle = plt.Circle((i, j * (max_neurons / max(1, size))), 0.2, fill=True, color='lightblue')
                ax_net.add_patch(circle)
            
            # Add connections if not the last layer
            if i < len(layer_sizes) - 1:
                for j in range(size):
                    for k in range(layer_sizes[i+1]):
                        ax_net.plot([i, i+1], 
                                  [j * (max_neurons / max(1, size)), 
                                   k * (max_neurons / max(1, layer_sizes[i+1]))], 
                                  'gray', alpha=0.3)
        
        ax_net.set_xlim(-0.5, len(layer_sizes) - 0.5)
        ax_net.set_ylim(-0.5, max_neurons - 0.5)
        ax_net.set_title("Neural Network Architecture (Simplified View)")
        st.pyplot(fig_net)

# --- Prediction Interface ---
st.subheader("🔮 Make a Prediction")
sun = st.slider("Sunlight (hours)", 0.0, 10.0, 5.0)
water = st.slider("Water (liters)", 0.0, 5.0, 2.5)

# Add temperature input if enabled
if st.session_state.use_temperature:
    temperature = st.slider("Temperature (°C)", 5.0, 35.0, 25.0)
    input_val = np.array([[sun / 10, water / 5, (temperature - 5) / 30]])
else:
    temperature = 25  # Default
    input_val = np.array([[sun / 10, water / 5]])

if st.session_state.trained:
    pred = st.session_state.nn.predict(input_val)[0][0]
    # Get the actual value from the realistic model
    true_val = realistic_plant_growth(sun, water, temperature)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🌿 Predicted Growth: **{pred:.4f}**")
    with col2:
        st.info(f"🔬 True Growth (Formula): **{true_val:.4f}**")
else:
    st.warning("Train the network to see predictions.")

# --- Additional Visualizations ---
if st.session_state.trained:
    st.subheader("📈 Additional Graphs for Deeper Insight")

    pred_all = st.session_state.nn.predict(X)
    
    # Calculate R² score with current model
    r2 = 1 - np.sum((y - pred_all) ** 2) / np.sum((y - y.mean()) ** 2)
    
    # For R² history, use a simulated improvement curve based on loss history
    if st.session_state.history:
        losses = np.array(st.session_state.history)
        # Convert loss history to approximate R² history
        r2_start = 0.1  # Starting R² value
        loss_improvement = (losses[0] - losses) / (losses[0] + 1e-10)  # Normalized improvement
        r2_history = r2_start + (r2 - r2_start) * (loss_improvement ** 0.5)  # Square root for non-linear improvement
        # Ensure R² is within valid range
        r2_history = np.clip(r2_history, 0, 1)
        
        # Sample to reduce points if history is very long
        sample_rate = max(1, len(r2_history) // 100)
        sampled_epochs = list(range(0, len(r2_history), sample_rate))
        sampled_r2 = r2_history[sampled_epochs]
    else:
        sampled_epochs = []
        sampled_r2 = []
    
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
        fig2, ax2 = plt.subplots()
        ax2.bar(["Model Prediction", "True (Formula)"], [pred, true_val], color=["orange", "green"])
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Growth Value")
        ax2.set_title("User Input Prediction vs True Value")
        st.pyplot(fig2)

    # 3. R² Accuracy Over Time (if available)
    if len(sampled_epochs) > 0:
        st.markdown("### 📊 R² Score Progress")
        fig3, ax3 = plt.subplots()
        ax3.plot(sampled_epochs, sampled_r2, color="darkred", linewidth=2)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Estimated R² Score")
        ax3.set_title("Approximate Accuracy (R²) Over Training")
        ax3.set_ylim(0, 1)
        st.pyplot(fig3)

    # 4. Decision Surface visualization (only for 2D input)
    if not st.session_state.use_temperature:
        st.markdown("### 🌐 Decision Surface")
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        
        try:
            # Create a mesh grid for visualization
            sun_range = np.linspace(0, 1, 40)  # Reduced grid size for better performance
            water_range = np.linspace(0, 1, 40)
            sun_grid, water_grid = np.meshgrid(sun_range, water_range)
            
            # Reshape for prediction
            X_mesh = np.c_[sun_grid.ravel(), water_grid.ravel()]
            
            # Make predictions
            Z = st.session_state.nn.predict(X_mesh).reshape(sun_grid.shape)
            
            # Plot contour
            contour = ax4.contourf(sun_grid, water_grid, Z, cmap='viridis', alpha=0.8, levels=15)
            plt.colorbar(contour, ax=ax4)
            
            # Plot training data points (sample to avoid overplotting)
            sample_idx = np.random.choice(range(len(X)), min(500, len(X)), replace=False)
            scatter = ax4.scatter(X[sample_idx, 0], X[sample_idx, 1], c=y[sample_idx].flatten(), 
                                cmap='coolwarm', edgecolor='k', s=30, alpha=0.7)
            
            ax4.set_xlabel('Normalized Sunlight')
            ax4.set_ylabel('Normalized Water')
            ax4.set_title('Neural Network Decision Surface')
            
            st.pyplot(fig4)
        except Exception as e:
            st.error(f"Could not generate decision surface: {str(e)}")
            st.info("This can happen with complex network architectures. Try simplifying the network.")
    else:
        st.info("Decision surface visualization is only available when using 2 input features (without temperature).")
        
        # Show feature importance instead
        st.markdown("### 🔍 Feature Importance Analysis")
        
        try:
            # Simple sensitivity analysis
            base_input = np.array([[sun/10, water/5, (temperature-5)/30]])
            base_pred = st.session_state.nn.predict(base_input)[0][0]
            
            # Test sensitivity to each input
            delta = 0.1
            sensitivities = []
            
            # Sunlight sensitivity
            test_input = base_input.copy()
            test_input[0, 0] += delta
            sens_sun = abs(st.session_state.nn.predict(test_input)[0][0] - base_pred) / delta
            
            # Water sensitivity
            test_input = base_input.copy()
            test_input[0, 1] += delta
            sens_water = abs(st.session_state.nn.predict(test_input)[0][0] - base_pred) / delta
            
            # Temperature sensitivity
            test_input = base_input.copy()
            test_input[0, 2] += delta
            sens_temp = abs(st.session_state.nn.predict(test_input)[0][0] - base_pred) / delta
            
            # Create bar chart of sensitivities
            fig5, ax5 = plt.subplots()
            features = ['Sunlight', 'Water', 'Temperature']
            sensitivities = [sens_sun, sens_water, sens_temp]
            colors = ['#FF9966', '#66B2FF', '#99CC99']
            
            bars = ax5.bar(features, sensitivities, color=colors)
            ax5.set_ylabel('Sensitivity Score')
            ax5.set_title('Feature Importance (Sensitivity Analysis)')
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax5.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            st.pyplot(fig5)
        except Exception as e:
            st.error(f"Could not perform sensitivity analysis: {str(e)}")

st.markdown("---")
st.markdown("Made with ❤️ for students learning about neural nets. Fine-tune, visualize, and play around to understand how training affects the network.")