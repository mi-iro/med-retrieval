cd med-qdrant
chmod 777 ./qdrant-1.13.6
nohup ./qdrant-1.13.6 --config-path ./config.yaml > ../recall.log &
cd ..