import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def kl_divergence(m1, s1, m2, s2):
    # D_KL(P || Q)
    return np.log(s2/s1) + (s1**2 + (m1 - m2)**2) / (2 * s2**2) - 0.5

st.title("Gaussian KL Divergence Visualizer")

# Sidebar for controls
st.sidebar.header("Parameters for P (Dist 1)")
m1 = st.sidebar.slider("Mean P", -5.0, 5.0, 0.0)
s1 = st.sidebar.slider("Std P", 0.1, 5.0, 1.0)

st.sidebar.header("Parameters for Q (Dist 2)")
m2 = st.sidebar.slider("Mean Q", -5.0, 5.0, 2.0)
s2 = st.sidebar.slider("Std Q", 0.1, 5.0, 1.0)

# Calculations
x = np.linspace(-10, 10, 1000)
y1 = (1 / (s1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m1) / s1)**2)
y2 = (1 / (s2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m2) / s2)**2)
kl_val = kl_divergence(m1, s1, m2, s2)

# Plotting
fig, ax = plt.subplots()
ax.plot(x, y1, label='P', color='#1f77b4', lw=2)
ax.fill_between(x, y1, alpha=0.3, color='#1f77b4')
ax.plot(x, y2, label='Q', color='#ff7f0e', lw=2)
ax.fill_between(x, y2, alpha=0.3, color='#ff7f0e')

ax.set_ylim(0, 1)
ax.legend()
ax.set_title(f"KL Divergence D(P||Q): {kl_val:.4f}")

# Display in Streamlit
st.pyplot(fig)

st.info(f"""
**Interpretation:** The KL Divergence measures how much information is lost when using distribution Q to approximate P. 
Note that if you swap the parameters of P and Q, the value will likely change!
""")