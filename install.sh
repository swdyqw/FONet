echo "****************** Installing pytorch ******************"
# conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

echo ""
echo ""
echo "****************** Installing yaml ******************"
# conda install PyYAML

echo ""
echo ""
echo "****************** Installing easydict ******************"
# conda install easydict
pip --default-timeout=100 install easydict -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

echo ""
echo ""
echo "****************** Installing cython ******************"
# conda install cython

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
# conda install opencv-python
pip --default-timeout=100 install opencv-python -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

echo ""
echo ""
echo "****************** Installing pandas ******************"
# conda install pandas

echo ""
echo ""
echo "****************** Installing tqdm ******************"
# conda install -y tqdm

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
# conda install pycocotools
pip --default-timeout=100 install pycocotools -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
# conda install jpeg4py
pip --default-timeout=100 install jpeg4py -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
# conda install tb-nightly
pip --default-timeout=100 install tb-nightly -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
# conda install tikzplotlib
pip --default-timeout=100 install tikzplotlib -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

echo ""
echo ""
echo "****************** Installing thop tool for FLOPs and Params computing ******************"
# conda install thop-0.0.31.post2005241907
pip --default-timeout=100 install thop-0.0.31.post2005241907 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

echo ""
echo ""
echo "****************** Installing colorama ******************"
# conda install colorama

echo ""
echo ""
echo "****************** Installing lmdb ******************"
# conda install lmdb

echo ""
echo ""
echo "****************** Installing scipy ******************"
# conda install scipy

echo ""
echo ""
echo "****************** Installing visdom ******************"
# conda install visdom
pip --default-timeout=100 install visdom -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com


echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
# conda install tensorboardX


# echo ""
# echo ""
# echo "****************** Downgrade setuptools ******************"
# conda install setuptools==59.5.0


# echo ""
# echo ""
# echo "****************** Installing wandb ******************"
# conda install wandb

# echo ""
# echo ""
# echo "****************** Installing timm ******************"
# pip install timm

echo ""
echo ""
echo "****************** Installation complete! ******************"
