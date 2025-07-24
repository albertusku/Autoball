#Install python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt

#Install c+++ dependencies
sudo apt-get update
sudo apt install libgpiod-dev


