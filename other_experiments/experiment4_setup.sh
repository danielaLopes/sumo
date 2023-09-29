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

echo "Downloading OSTrain, OSValidate, and OSTest extracted features in DeepCoFFEA's format ..."
curl -o sumo_features_for_deepcoffea.tar.gz -L "https://zenodo.org/record/8386335/files/sumo_features_for_deepcoffea.tar.gz?download=1"
echo "Extracting sumo_features_for_deepcoffea.tar.gz"
tar -xfv sumo_features_for_deepcoffea.tar.gz

echo "Converting SUMo's features into DeepCoFFEA's features ..."
python3 sumo_to_deepcoffea_features.py

# TODO: Sill producing errors
#echo "Generating model files for DeepCoFFEA ..."
#./scripts/run_dcf_20230521.sh 
#./scripts/run_dcf_20230521_2.sh 