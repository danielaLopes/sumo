cd dl_comparisons
mkdir -p datasets
# Download DeepCoFFEA's dataset
echo "Downloading DeepCoFFEA's data ..."
cd datasets
if [ ! -d "CrawlE_Proc" ]; then
    wget -O CrawlE_Proc.zip "https://drive.google.com/u/0/uc?id=1ZYFXfESD15SAR4Q8hsoVYdTHpTD8Orys&export=download&confirm=yes"
    unzip CrawlE_Proc.zip
fi
if [ ! -d "deepcoffea" ]; then
    git clone https://github.com/traffic-analysis/deepcoffea.git
fi

echo "Converting SUMo's features into DeepCoFFEA's features ..."
python3 sumo_to_deepcoffea_features.py
