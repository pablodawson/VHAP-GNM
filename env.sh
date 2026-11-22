# Source this after `conda activate <env>` if a global CUDA_HOME (e.g. from ~/.bashrc) points at another env's
# CUDA toolkit. It makes JIT-compiled extensions (nvdiffrast) use the active env's toolkit. See README > Troubleshooting.
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CPATH"
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"
