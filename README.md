## SaliencyELD
Source code for our CVPR 2016 paper "Deep Saliency with Encoded Low level Distance Map and High Level Features" by [Gayoung Lee](https://sites.google.com/site/gylee1103/), [Yu-Wing Tai](www.gdriv.es/yuwing) and [Junmo Kim](https://sites.google.com/site/siitkaist/professor).

Please contact to gylee1103@gmail.com if you have any question about the implementation and the installation of the code.

Acknoledgement : Our code uses [Caffe](http://github.com/BVLC/caffe) and [VLfeat](http://www.vlfeat.org) library.

## Usage
1. **Dependensies**
    1) OS : Our code is tested on Ubuntu 14.04
    2) Caffe : Caffe that we used is contained in this repository.
    3) VLFeat
    4) OpenCV 3.0 : We used OpenCV 3.0, but the code may work with OpenCV 2.4.X version.
    5) g++ : Our code needs openmp and c++11 and was tested with 4.9.2.
    6) Boost
    7) CMake
2. **Installation**
    1) Get our pretrained model and VGG16 model. NOTE: Some paths for caffe models and prototxts are hard-coded in **main.cpp**. Check them if you download models in the other folder.
        ```
        cd $(PROJECT_ROOT)/models/
        sh get_models.sh
        ```
    1) Build Caffe in the project folder using CMake:
        ```
        cd $(PROJECT_ROOT)/caffe/
        mkdir build
        cd build/
        cmake ..
        make -j4
        ```
    2) Change library paths in $(PROJECT_ROOT)/CMakeLists.txt for your custom environment and build our code:
        ```
        cd $(PROJECT_ROOT)
        edit CMakeList.txt
        mkdir build
        cd build/
        cmake ..
        make
        ```
    3) Run the executable file which takes one argument for the path of the directory containing test images:
        ```
        ./SaliencyELD ../test_images
        ```
    4) The results will be generated in the test directory.
    
## Citing our work
Please kindly cite our work if it helps your research:
    @inproceedings{lee2016saliency,
        title = {Deep Saliency with Encoded Low level Distance Map and High Level Features},
        author={Gayoung, Lee and Yu-Wing, Tai and Junmo, Kim},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2016}
    }
