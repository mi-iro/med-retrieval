cd med-qdrant-scripts
export CUDA_VISIBLE_DEVICES=4,5,6,7
nohup python api.py 10002 > ../rerank.log &
cd ..