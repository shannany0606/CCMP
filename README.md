# CCMP: Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction (CVPR 2026)

## News
- [2026/2/27] Data, models and codes are released. We provide extensive preprocessed resources to simplify the pipeline and facilitate the reproduction of our work.
- [2026/2/24] Our paper is available on [arXiv](https://arxiv.org/abs/2602.18996).
- [2026/2/21] Our paper is accepted by CVPR 2026. Thanks to all co-authors!

## 1. Installation

```
conda create -n ccmp python=3.11 -y
conda activate ccmp
pip install -r requirements.txt
pip install xformers==0.0.31.post1 --no-deps
```


## 2. Model and Data Preparation

**Models.** Download the pretrained checkpoints `dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth` and `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` from the official [DINOv3](https://github.com/facebookresearch/dinov3) repository, and place them under `SegSwap/model`.

**Ego-Exo4D Data.** The standard data preparation pipeline follows [SegSwap](https://github.com/EGO4D/ego-exo4d-relation/tree/main/correspondence/SegSwap) to download the Ego-Exo4D videos and preprocess them into image sequences. 

Since this procedure is time-consuming, we provide the preprocessed dataset we used, `true_data.zip`, available at [Baidu Netdisk](https://pan.baidu.com/s/1aOSl6AlREePafaXSl8KS4Q?pwd=fs8x). Simply unzip the archive under `SegSwap`. (**Note:** This dataset may be smaller than the one obtained via the standard pipeline. Please refer to this [issue](https://github.com/EGO4D/ego-exo4d-relation/issues/3) for details.)

**HANDAL-X Data.** Please follow the instructions in [ObjectRelator](https://github.com/lovelyqian/ObjectRelator/blob/main/docs/DATASET.md) to prepare the dataset and place it under `SegSwap/handal`.

To simplify the pipeline, you may skip the JSON generation step by directly downloading our pre-generated annotation files, `handal_train_visual.json` and `handal_test_visual.json`, from [Baidu Netdisk](https://pan.baidu.com/s/1aOSl6AlREePafaXSl8KS4Q?pwd=fs8x).


## 3. Training 

``` Bash
cd SegSwap/train
bash run.sh
```

We release our pretrained model `best_test_miou.pth` at [Baidu Netdisk](https://pan.baidu.com/s/1aOSl6AlREePafaXSl8KS4Q?pwd=fs8x), you can use it for quick inference.

## 4. Inference

For Ego2Exo task,
``` Bash
cd SegSwap/train
bash run_ego.sh
```

For Exo2Ego task,
``` Bash
cd SegSwap/train
bash run_exo.sh
```

the above command should produce a `ego-exo_test_results_ttt.json`/`exo-ego_test_results_ttt.json`/`ego-exo_test_results.json`/`exo-ego_test_results.json` file which can be then be used to run evaluation. We provide our inference results `exo-ego_test_results_ttt.json` and `exo-ego_test_results_ttt.json` at [Baidu Netdisk](https://pan.baidu.com/s/1aOSl6AlREePafaXSl8KS4Q?pwd=fs8x).

## 5. Evaluation

**Ego-Exo4D Benchmark**

To run the evaluation, first process the annotations
``` Bash
cd evaluation
python process_annotations.py --data_path ../SegSwap/true_data --annotations_path /data/egoexo/annotations/relations_test.json --split test --output_path ../SegSwap/output/correspondence-gt.json
```

To simplify the pipeline, you may directly downloading our pre-generated annotation files `correspondence-gt.json` from [Baidu Netdisk](https://pan.baidu.com/s/1aOSl6AlREePafaXSl8KS4Q?pwd=fs8x).

then run the following command:

for Ego2Exo task,
```Bash
cd evaluation
python3 evaluate_egoexo.py --gt-file ../SegSwap/output/correspondence-gt.json --pred-file ../SegSwap/output/1111_dinov3cnlarge_dinov3large_dice5_bs16as16ep200_mxlr1e5_lp20lr1e4_tttlayers4_iter2/ego-exo_test_results_ttt.json
```

for Exo2Ego task,
```Bash
cd evaluation
python3 evaluate_exoego.py --gt-file ../SegSwap/output/correspondence-gt.json --pred-file ../SegSwap/output/1110_dinov3cnlarge_dinov3large_dice5_bs16as16ep200_mxlr1e5_lp20lr1e4_tttlayers11_iter6/exo-ego_test_results_ttt.json
```

**HANDAL-X Benchmark**

```Bash
cd SegSwap 
torchrun --nproc_per_node=8 eval_handal.py --json_path handal/handal_test_visual.json --model_path train/output/1102_dinov3cnlarge_dinov3large_dice5_bs16as16ep200_mxlr1e5_lp20lr1e4/best_test_miou.pth --root_path handal --image_size 512 --backbone_size large --backbone_type dinov3 --extractor_type dinov3_cn_large --use_amp --dist
```

## Citation

If you think our work is useful for your research, please use the following BibTeX entry.
```
@article{yan2026learning,
  title={Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction},
  author={Yan, Shannan and Zheng, Leqi and Lv, Keyu and Ni, Jingchen and Wei, Hongyang and Zhang, Jiajun and Wang, Guangting and Lyu, Jing and Yuan, Chun and Rao, Fengyun},
  journal={arXiv preprint arXiv:2602.18996},
  year={2026}
}
```

The code in this repo is based on [XSegTx](https://github.com/EGO4D/ego-exo4d-relation/tree/main/correspondence/SegSwap), [DINOv3](https://github.com/facebookresearch/dinov3) and [DINOv2](https://github.com/facebookresearch/dinov2).