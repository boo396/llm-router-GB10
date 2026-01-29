FROM nvcr.io/nvidia/tritonserver:25.12-py3

# RUN git clone https://github.com/triton-inference-server/python_backend -b r24.10
COPY src/router-server/requirements.txt /tmp/requirements.txt
RUN pip install --break-system-packages -r /tmp/requirements.txt
# ENV NVIDIA_API_KEY=nvapi-YOUR-KEY-HERE