conda create -n analysis_review python=3.11 -y
conda activate analysis_review
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt