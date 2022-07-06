### General Policy Evaluation and Improvement by Learning to Identify Few But Crucial States

[arxiv](https://arxiv.org/abs/2207.01566)

Install requirements: 
```bash
pip install -r requirements.txt
```

Reproduce PSSVF results for the MuJoCo environments:
```bash
python3 pssvf.py --env_name Ant-v3 --use_gpu 1
python3 pssvf.py --env_name HalfCheetah-v3 --use_gpu 1
python3 pssvf.py --env_name Swimmer-v3 --use_gpu 1
python3 pssvf.py --env_name Walker2d-v3 --use_gpu 1
python3 pssvf.py --env_name Hopper-v3 --use_gpu 1
python3 pssvf.py --env_name InvertedDoublePendulum-v2 --use_gpu 1
```
