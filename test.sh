echo 'Segment detection ...'
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/SLCO_kbref.py --checkpoint ./work_dirs/SLCO_kbref/checkpoint0068.pth --batch_size_test 32 --test_split val --only_segment
echo 'Prompt-based retrieval ...'
CUDA_VISIBLE_DEVICES=0 python ./knowledge_retrieval/prompt_based_retrieval.py --lm bert --bmn bert-large-uncased --bmd ./knowledge_retrieval/data/plm-checkpoint-125120
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/SLCO_kbref.py --checkpoint ./work_dirs/SLCO_kbref/checkpoint0068.pth --batch_size_test 32 --test_split val --know_retrieval_results ./knowledge_retrieval/knowledge_retrieval_results.json