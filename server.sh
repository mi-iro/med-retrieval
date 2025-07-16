export PATH=/opt/conda/bin:$PATH
export PATH=/mnt/public/data/lh/yqj/miniconda3/bin/conda:$PATH
conda init
exec bash
cd /mnt/public/lianghao/wzr/med_reseacher/med-retrieval
conda activate /mnt/public/lianghao/wzr/r2med_env
source start_main.sh
source start_rerank.sh
source start_recall.sh
# sleep 999999999999999