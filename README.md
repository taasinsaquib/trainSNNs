# trainSNNs
Code to train models for thesis 

# Connect Local Runtime to Google Colab

* `pip install --upgrade jupyter_http_over_ws>=0.0.7 && jupyter serverextension enable --py jupyter_http_over_ws`
* `jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0`

# Run Tensorboard

* https://www.tensorflow.org/tensorboard/get_started
* `tensorboard --logdir=runs`

# Setup PyTorch for Local GPU

* install CUDA (correct version for the GPU you have)
* https://pytorch.org/get-started/locally/