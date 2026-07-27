# Start from the official Dolfinx image
FROM dolfinx/dolfinx:stable

# ---------------------------
# Upgrade pip and Python packages
# ---------------------------
RUN python3 -m pip install --upgrade pip


# Install scientific Python stack compatible with JAX
RUN pip install --no-cache-dir \
    "numpy<2" \
    "scipy<2" \
    pandas \
    tqdm \
    scikit-learn \
    matplotlib \
    matplotlib-label-lines 
    
RUN pip install --no-cache-dir \
    jax==0.4.25 \
    jaxlib==0.4.25+cuda11.cudnn86 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
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
