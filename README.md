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

# ffmpeg

* convert images to video
* `C:\Users\taasi\Downloads\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin`
* `ffmpeg.exe -i C:\Users\taasi\Desktop\biomechanical_eye_siggraph_asia_19\dump\image%06d.bmp -vcodec libx264 -crf 25  -pix_fmt yuv420p C:\Users\taasi\trainSNNs\verifyModels\test.mp4`