# Start from the official Dolfinx image
FROM --platform=linux/amd64 dolfinx/dolfinx:stable


# ---------------------------
# Upgrade pip and Python packages
# ---------------------------
RUN python3 -m pip install --upgrade pip


# Install scientific Python stack compatible with JAX
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    "pandas<3" \
    seaborn \
    tqdm \
    scikit-learn \
    matplotlib \
    matplotlib-label-lines 
    
RUN pip install --no-cache-dir \
    jax \
    jaxlib \
    meshio \
    pyfiglet \
    jax-fem \
    pyyaml \
    optax \
    torch==2.6.0 \
    gpytorch==1.14 \
    normflows==1.7.3 \
    openpyxl==3.1.5 \
    SALib==1.5.1


# Set working directory
WORKDIR /home/mmdiscovery/shared

CMD ["bash"]
