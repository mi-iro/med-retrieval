cd med-qdrant-qwen
chmod 777 ./qdrant-1.13.6
nohup ./qdrant-1.13.6 --config-path ./config.yaml > ../recall-qwen.log &
cd ..