data_path=${1:-'.'}
mkdir -p $data_path

wget -P $data_path https://cernbox.cern.ch/index.php/s/SAHcpBDTp4P9jfO/download
tar -xzf $data_path/download