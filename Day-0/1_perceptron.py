import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


### Use "streamlit run 1_perceptron.py" to run the app  ###

# --- Helper Function ---
def step_function(z):
    return 1 if z > 0 else 0

# --- Improved Styling with Seaborn ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12

# --- Streamlit App ---
st.title("Perceptron Playground: Can You Solve AND?")
st.subheader("Tweak the Neuron's Brain to Classify!")

# --- Sidebar for Manual Control ---
st.sidebar.header("Neuron Brain Controls")
initial_w1 = st.sidebar.slider("Weight for Input 1 (w1)", -3.0, 3.0, 1.0, 0.1)
initial_w2 = st.sidebar.slider("Weight for Input 2 (w2)", -3.0, 3.0, 1.0, 0.1)
initial_bias = st.sidebar.slider("Bias (b)", -3.0, 3.0, -1.5, 0.1)

# --- Data Points ---
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 0, 1])
point_labels = ["(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"]
colors = np.array(['skyblue', 'skyblue', 'skyblue', 'salmon']) # More visually distinct

# --- Prediction Logic ---
def predict(X, w1, w2, bias):
    z = (w1 * X[0]) + (w2 * X[1]) + bias
    return step_function(z)

# --- Visualization Area (Interactive Plot) ---
st.subheader("AND Gate World and Your Neuron's Thinking")
fig, ax = plt.subplots()

# Scatter plot of data points with better styling
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=colors[y_train], s=300, marker='o', edgecolors='black', linewidths=0.7)
for i, label in enumerate(point_labels):
    ax.annotate(label, (X_train[i, 0] + 0.05, X_train[i, 1] + 0.05), fontsize=10, alpha=0.7)

ax.set_xlabel("Input 1 (x1)", fontsize=14)
ax.set_ylabel("Input 2 (x2)", fontsize=14)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_aspect('equal', adjustable='box')

# Plot the decision boundary with more visual emphasis
w1, w2, bias = initial_w1, initial_w2, initial_bias
if w2 != 0:
    x_boundary = np.linspace(-0.5, 1.5, 100)
    y_boundary = (-w1 / w2) * x_boundary - (bias / w2)
    ax.plot(x_boundary, y_boundary, 'lime', linewidth=3, label="Neuron's Decision Line")
    ax.fill_between(x_boundary, y_boundary, 1.5, color='lightcoral', alpha=0.2, label='Predicts 1')
    ax.fill_between(x_boundary, y_boundary, -0.5, color='lightblue', alpha=0.2, label='Predicts 0')
    ax.legend(fontsize=12)
elif w1 != 0:
    y_boundary = np.linspace(-0.5, 1.5, 100)
    x_boundary = -bias / w1 * np.ones_like(y_boundary)
    ax.plot(x_boundary, y_boundary, 'lime', linewidth=3, label="Neuron's Decision Line")
    ax.fill_betweenx(y_boundary, x_boundary, 1.5, color='lightcoral', alpha=0.2, label='Predicts 1')
    ax.fill_betweenx(y_boundary, x_boundary, -0.5, color='lightblue', alpha=0.2, label='Predicts 0')
    ax.legend(fontsize=12)
else:
    ax.axhline(y=-bias, color='lime', linewidth=3, label="Neuron's Decision Line (Horizontal)")
    ax.fill_between(np.linspace(-0.5, 1.5, 100), -bias, 1.5, color='lightcoral', alpha=0.2, label='Predicts 1')
    ax.fill_between(np.linspace(-0.5, 1.5, 100), -bias, -0.5, color='lightblue', alpha=0.2, label='Predicts 0')
    ax.legend(fontsize=12)

st.pyplot(fig)

# --- Neuron Ball with Internal Values ---
st.subheader("Inside the Neuron Ball")
st.write("See the calculations happening inside as you adjust the controls.")

neuron_fig_inner, neuron_ax_inner = plt.subplots(figsize=(6, 6))
circle = patches.Circle((0.5, 0.5), 0.4, color='gold', alpha=0.6, edgecolor='black')
neuron_ax_inner.add_patch(circle)
neuron_ax_inner.axis('off')

# Display weights and bias inside the neuron
neuron_ax_inner.text(0.5, 0.6, f"w1 = {initial_w1:.2f}", ha='center', fontsize=14)
neuron_ax_inner.text(0.5, 0.45, f"w2 = {initial_w2:.2f}", ha='center', fontsize=14)
neuron_ax_inner.text(0.5, 0.3, f"b = {initial_bias:.2f}", ha='center', fontsize=14)
neuron_ax_inner.text(0.5, 0.8, "Neuron Core", ha='center', fontsize=16, fontweight='bold')

st.pyplot(neuron_fig_inner)

# --- Detailed Output and Explanation ---
st.subheader("Decoding the Neuron's Decision")

selected_point_index = st.selectbox("Pick an input to analyze:", range(len(X_train)), format_func=lambda i: f"{point_labels[i]} (Target: {y_train[i]})")
selected_input = X_train[selected_point_index]
target = y_train[selected_point_index]
prediction = predict(selected_input, initial_w1, initial_w2, initial_bias)

st.info(f"Analyzing input: **x1 = {selected_input[0]}, x2 = {selected_input[1]}**")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Weight 1 (w1)", f"{initial_w1:.2f}")
with col2:
    st.metric("Weight 2 (w2)", f"{initial_w2:.2f}")
with col3:
    st.metric("Bias (b)", f"{initial_bias:.2f}")

weighted_sum = (initial_w1 * selected_input[0]) + (initial_w2 * selected_input[1]) + initial_bias
st.write(f"**1. Weighted Sum (z):** z = ({initial_w1:.2f} * {selected_input[0]}) + ({initial_w2:.2f} * {selected_input[1]}) + ({initial_bias:.2f}) = **{weighted_sum:.2f}**")

st.write("**2. Activation (Step Function):**")
if weighted_sum > 0:
    st.success(f"Since z = {weighted_sum:.2f} > 0, the neuron activates and outputs **1**.")
else:
    st.warning(f"Since z = {weighted_sum:.2f} ≤ 0, the neuron does not activate and outputs **0**.")

st.subheader(f"Neuron's Verdict for {point_labels[selected_point_index]}:")
if prediction == target:
    st.balloons()
    st.success(f"✅ Correct! Your neuron predicted **{prediction}**, and the target was **{target}**.")
else:
    st.error(f"❌ Incorrect! Your neuron predicted **{prediction}**, but the target was **{target}**.")

st.info("Keep tweaking the weights and bias in the sidebar to see if you can get the 'Neuron's Decision Line' to perfectly separate the blue (0) points from the red (1) point. The goal is to classify all inputs correctly!")