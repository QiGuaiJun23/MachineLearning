# Tensorflow环境配置

本地配置

Virtualenv安装

    pip install -U virtualenv

    virtualenv --system-site-packages -p python3 ./tf_py3

    source tf_py3/bin/activate

    pip install tensorflow


云配置

    GPU版本

        需要安装CUDA驱动 

        nvidia-smi

    接下来的步骤如上

    tf.test.is_gpu_available()#查看是否可以用GPU

pip与pip3的安装不一样，pip3是python3的安装
