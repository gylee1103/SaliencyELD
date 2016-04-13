#!bin/bash
mkdir -p ELD
mkdir -p VGG16
cd VGG16/
if [ ! -f VGG_ILSVRC_16_layers.caffemodel ]; then
  wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel -O VGG_ILSVRC_16_layers.caffemodel
fi
cd ../ELD
if [ ! -f eldmodel_iter_112192.caffemodel ]; then
  wget https://www.dropbox.com/s/6gqoj2lgz177hbr/eldmodel_iter_112192.caffemodel -O eldmodel_iter_112192.caffemodel
fi
cd ..


