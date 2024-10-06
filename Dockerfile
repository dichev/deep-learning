FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime


# Requred by torch.compile()
RUN apt-get update && apt-get install -y g++


# Using pip simply because it is much faster than the conda solver
RUN pip install --no-cache-dir matplotlib numpy openai pandas pytest regex scikit-learn scipy tqdm gdown fsspec
RUN pip install --no-cache-dir networkx tensorboard datasets torch_geometric einops tiktoken torcheval pybars3


WORKDIR /deep-learning
COPY . .
ENV PYTHONPATH="/deep-learning:/deep-learning/src:/deep-learning/test"

CMD ["sh", "-c", "\
    pytest ./test || true && \
    echo 'Examples:' && find examples/ -type f -name '*.py' | sed 's|^|  python |' && \
    echo 'The container is kept alive..' && \
    tail -f /dev/null \
"]
