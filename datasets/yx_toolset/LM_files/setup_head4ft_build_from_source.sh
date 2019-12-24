#!/bin/sh

work_dir=$(pwd)

# for yum-config-manager
yum -y install epel-release wget tree yum-utils net-tools centos-release-scl

# 使用国内YUM镜像源
mv /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.orig
wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo

# 添加Intel MKL安装源
yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB

# FFMPEG安装源：国外服务器，速度慢，暂不安装，除非有视频流处理需要
#rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
#rpm -Uh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm

# Nvidia安装源，安装驱动、cuda等
#rpm -Uh http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-7.0-28.x86_64.rpm 

yum clean all

# 使用国内PIP镜像源
mkdir -p ~/.pip/
cat > ~/.pip/pip.conf << EOF
[global]
index-url = http://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host=mirrors.aliyun.com
EOF

# 更新工具链
yum -y install devtoolset-8 gcc-gfortran llvm-devel
yum -y install python2-devel python2-tools python2-pip
yum -y install python3-devel python3-tools python3-pip
# automake, etc
yum -y install openssl-devel bzip2 git htop unzip curl-devel autoconf automake libtool gperftools-devel
# 安装MKL加速库
yum -y install graphviz doxygen atlas-devel intel-mkl-64bit-2019.4-070 #intel-mkl-64bit-2017.4-061
# 编译caffe依赖包
#yum install boost-devel leveldb-devel hdf5-devel glog-devel gflags-devel lmdb-devel

# nvidia cuda支持
#yum search cuda
#yum install cuda

# 每次登录时，新的工具链生效
ln -snf /opt/rh/devtoolset-8/enable /etc/profile.d/devtoolset.sh; rdate=2019
cat > /etc/profile.d/intel_mkl.sh << EOF
. /opt/intel/compilers_and_libraries_$rdate/linux/bin/compilervars.sh -arch intel64 -platform linux
export PYTHONPATH=\$PYTHONPATH:/opt/yxtech/python/traffic-sign1:/opt/yxtech/python/traffic-sign0
export KMP_AFFINITY=granularity=fine,compact,1,0
export vCPUs=\$(cat /proc/cpuinfo | grep processor | wc -l)
export OMP_NUM_THREADS=\$((vCPUs / 2))
EOF
chmod a+x /etc/profile.d/{intel_mkl.sh,devtoolset.sh}

echo ""
echo "Extracting Installer..."
echo ""

mkdir -p /opt/yxtech/{model,python,cpp,misc}

export _tmpdir=/opt/yxtech/misc
#export _tmpdir=$(mktemp -d /tmp/tlpr.XXXXXX)

ARCHIVE=`awk '/^__ARCHIVE_BELOW__/ {print NR + 1; exit 0; }' $0`

tail -n+$ARCHIVE $0 | tar xjf - -C $_tmpdir

chmod -R a+rw $_tmpdir && cd $_tmpdir/installer

# 强制引入工具链环境
. /etc/profile.d/devtoolset.sh
. /etc/profile.d/intel_mkl.sh

pip2 install --upgrade pip
pip2 install numpy==1.16.4
pip2 install cython
pip2 install matplotlib==2.2.4 pycocotools pillow easydict pyyaml typing
pip2 install graphviz==0.8.4 requests==2.22.0

pip3 install --upgrade pip
rm -f /usr/bin/pip*3* /usr/local/bin/pip
ln -snr /usr/bin/pip /usr/local/bin/
ln -snf /usr/local/bin/pip3* /usr/bin/

pip3 install numpy
pip3 install cython
pip3 install matplotlib pycocotools pillow easydict pyyaml typing
pip2 install graphviz==0.8.4 requests==2.22.0

#ninja
cd ninja && ./configure.py --bootstrap
mv ninja /usr/bin/ && cd -

#cmake
tar xvf cmake-3.16.0-rc2.tar.gz && cd cmake-3.16.0-rc2
./configure --prefix=/usr
make -j${vCPUs} install && cd -

#openblas
mkdir -p OpenBLAS/build && cd OpenBLAS/build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr ..
make -j${vCPUs} install && cd -

#lapack
tar xvf lapack-3.8.0.tar.gz && mkdir -p lapack-3.8.0/build && cd lapack-3.8.0/build
cmake -DBUILD_SHARED_LIBS=ON -DLAPACKE=ON -DCMAKE_INSTALL_PREFIX=/usr ..
cmake --build . -j${vCPUs} --target install && cd -

#protobuf
cd protobuf && ./configure --prefix=/usr
make -j${vCPUs} install && ldconfig && cd -

#jemalloc
cd jemalloc && ./configure --disable-initial-exec-tls --prefix=/usr
make -j${vCPUs} install && ln -snf /usr/lib/libjemalloc.so /lib64/libjemalloc.so.2 && cd -

#mkl-dnn
tar xvf mkl-dnn-1.0.4.tar.bz2 && mkdir -p mkl-dnn-1.0.4/build
cd mkl-dnn-1.0.4/build && sed -i 's/!USE_MKL/USE_MKL/g' ../src/cpu/gemm/gemm_pack.cpp
cmake -DCMAKE_BUILD_TYPE=Release -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF -D_DNNL_USE_MKL=ON -DCMAKE_CXX_FLAGS=-fpermissive -DCMAKE_INSTALL_PREFIX=${MKLROOT} -DCMAKE_INSTALL_LIBDIR=lib/intel64 ..
make -j${vCPUs} install && cd -

#opencv -DWITH_CUDA=ON -DWITH_CUBLAS=ON -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON
tar xvf 3.4.8.tar.gz && mkdir -p opencv-3.4.8/build && cd opencv-3.4.8/build
sed -i 's/https.*/file:$ENV{_tmpdir}\/installer\"/g' ../3rdparty/ippicv/ippicv.cmake
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_SHARED_LIBS=ON -DOPENCV_GENERATE_PKGCONFIG=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -GNinja ..
ninja install && cd -

#anaconda
#_prog=Anaconda3-2019.07-Linux-x86_64.sh
#_conda_path=/opt/anaconda3
#wget https://repo.anaconda.com/archive/${_prog}; chmod a+x ${_prog}
#./${_prog} -bup ${_conda_path}
#. ${_conda_path}/bin/activate
#conda init

#conda containers
#yes y | conda create -n torch-cpu-python3.6 python=3.6
#yes y | conda create -n mxnet-cpu-python2.7 python=2.7

#mxnet on cpu
tar xvf mxnet.tar.bz2 && mkdir -p mxnet/build
#cd mxnet && make -j${vCPUs} USE_OPENCV=1 USE_BLAS=mkl USE_MKLDNN=1 MKL_ROOT=${MKLROOT} USE_INTEL_PATH=/opt/intel
#cd python && python setup.py install && cd ${_tmpdir}/installer
cd mxnet/build && cmake -DUSE_OPENCV=1 -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_CPP_PACKAGE=1 -DUSE_MKLDNN=1 -DUSE_MKL_IF_AVAILABLE=1 -DMKL_ROOT=${MKLROOT} -DUSE_BLAS=mkl -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${MKLROOT} -DCMAKE_INSTALL_LIBDIR=lib/intel64 -GNinja ..
ninja install && cp -fav 3rdparty/openmp/runtime/src/libomp.so ${MKLROOT}/lib/intel64/
cd ../python && python setup.py install && cd ${_tmpdir}/installer

#conda activate mxnet-cpu-python2.7
#pip2 install mxnet-mkl

#pytorch
#git clone --recursive https://gitee.com/mirrors/pytorch.git
#export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#python setup.py install

#conda activate torch-cpu-python3.6
pip3 install torch==1.0.1 torchvision==0.2.1

#TODO

mv -fv opt/yxtech/{model,python,cpp} /opt/yxtech/
# 分割目标识别模型评估工具，躺在地面上的较大目标
ln -snf /opt/yxtech/python/utilities/sign0_detector.py /usr/bin/sign0_detector
# 传统目标识别模型评估工具，有支撑物挂起来的符号目标
ln -snf /opt/yxtech/python/utilities/sign1_detector.py /usr/bin/sign1_detector
# 丰图标注数据转COCO格式工具
ln -snf /opt/yxtech/python/utilities/ft2coco_v1.1.py /usr/bin/ft2coco

cd $work_dir
#rm -rf $_tmpdir

echo ""
echo "Done"
echo ""

exit 0

__ARCHIVE_BELOW__
