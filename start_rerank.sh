cd med-qdrant-scripts
export CUDA_VISIBLE_DEVICES=2,3
nohup python api.py 10002 > ../rerank.log &
cd ..