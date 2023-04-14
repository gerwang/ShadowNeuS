# ShadowNeuS

Code for our CVPR 2023 paper "**ShadowNeuS: Neural SDF Reconstruction by Shadow Ray Supervision**", which draws inspiration from [NeRF](https://www.matthewtancik.com/nerf) and presents a new ray supervision scheme for reconstructing scenes from single-view shadows.

### [Project Page](https://gerwang.github.io/shadowneus/) | [Paper](https://arxiv.org/abs/2211.14086) | [Video](https://www.youtube.com/watch?v=ZZKWmPuzNWM) | [Dataset](https://drive.google.com/drive/folders/1Sr30kdvCD2tXNAONzcnF5xnoMXasylyA?usp=sharing)

https://user-images.githubusercontent.com/25758401/231767821-c03510b1-a087-4f7c-bdac-959b2466280d.mp4

## Usage

### Setup

```bash
git clone https://github.com/gerwang/ShadowNeuS.git
cd ShadowNeuS
conda create -n shadowneus python=3.9
conda activate shadowneus
pip install -r requirements.txt
```

### Testing

Here we show how to test our code on an example scene. Before testing, you need to

- Download the [example data](https://drive.google.com/file/d/1JD-O-VKkWz9_lBerEhqf3_ome34bijc_/view?usp=sharing) and unzip it to `./public_data/nerf_synthetic`.

- Download the pretrained checkpoint of `lego_specular_point` [here](https://drive.google.com/file/d/1zt3h0Jl3cb5v5T-Wv5GJjtSo6wA5JtEJ/view?usp=sharing) and unzip it to `./exp`.

#### Novel-view synthesis

```bash
python exp_runner.py --mode validate_view --conf confs/point_color.conf --case lego_specular_point --is_continue --data_sub 1 --test_mode
```

See the results at `./exp/lego_specular_point/point_color/novel_view/validations_fine/`.

#### Extracting mesh

```bash
python exp_runner.py --mode validate_mesh --conf confs/point_color.conf --case lego_specular_point --is_continue --data_sub 1 --test_mode
```

See the results at `./exp/lego_specular_point/point_color/meshes/00150000.ply`.

#### Relighting

```bash
python exp_runner.py --mode validate_relight --conf confs/point_color.conf --case lego_specular_point --is_continue --data_sub 1 --test_mode
```

See the results at `./exp/lego_specular_point/point_color/novel_light/validations_fine/`.

#### Material editing

```bash
python exp_runner.py --mode validate_relight_0_point_gold --conf confs/point_color.conf --case lego_specular_point --is_continue --test_mode
```

See the results at `./exp/lego_specular_point/point_color/novel_light_gold/validations_fine/`.

The `--mode` option can be ` validate_relight_<img_idx>_<light>_<material>`, where `<img_idx>` is the image index in the training dataset, `light` can be `point` or `dir` which determines whether a point light or direction light is used, and `material` can be `gold` or `emerald`.

#### Evaluate normal and depth map

```bash
python exp_runner.py --mode validate_normal_depth --conf confs/point_color.conf --case lego_specular_point --is_continue --data_sub 1 --test_mode
```

See the results at `./exp/lego_specular_point/point_color/quantitative_compare/`.

#### Environment relighting

```bash
python exp_runner.py --mode validate_env_0_0 --conf confs/point_color.conf --case lego_specular_point --is_continue --data_sub 1 --test_mode
python exp_runner.py --mode validate_env_0_0.25 --conf confs/point_color.conf --case lego_specular_point --is_continue --data_sub 1 --test_mode
python exp_runner.py --mode validate_env_0 --conf confs/point_color.conf --case lego_specular_point --is_continue --data_sub 1 --test_mode
python exp_runner.py --mode validate_env_0_0.75 --conf confs/point_color.conf --case lego_specular_point --is_continue --data_sub 1 --test_mode
```

Download environment maps of [Industrial Workshop Foundry](https://polyhaven.com/a/industrial_workshop_foundry), [Thatch Chapel](https://polyhaven.com/a/thatch_chapel), [Blaubeuren Night](https://polyhaven.com/a/blaubeuren_night) and [J&E Gray Park](https://polyhaven.com/a/je_gray_park). Extract them to `./public_data/envmap`.

```bash
python env_relight.py --work_path ./exp/lego_specular_point/point_color/ --env_paths ./public_data/envmap/industrial_workshop_foundry_4k.exr,./public_data/envmap/thatch_chapel_4k.exr,./public_data/envmap/blaubeuren_night_4k.exr,./public_data/envmap/je_gray_park_4k.exr --save_names super_workshop,super_chapel,super_night,super_park --super_sample --n_theta 128 --n_frames 128 --device_ids 0,1,2,3
```

See the results at `./exp/lego_specular_point/point_color/super_workshop`, `super_chapel`, `super_night`, and `super_park`. The above command is tested on four RTX 3090 GPUs.

Output:


https://user-images.githubusercontent.com/25758401/231927850-cd4ace70-b034-4aed-b1b4-8140c1338bec.mp4



### Training

You can download training data from [here](https://drive.google.com/drive/folders/1Sr30kdvCD2tXNAONzcnF5xnoMXasylyA?usp=sharing).

#### On point light RGB inputs

Extract [`point_light.zip`](https://drive.google.com/file/d/1Wo-0iNRYZs02GAfwlZ9SsFdUWjze6IZ-/view?usp=sharing) and move each scene to `./public_data/nerf_synthetic`. Then run

```bash
python exp_runner.py --mode train --conf ./confs/point_color.conf --case <case_name>_specular_point
```

#### On point light shadow inputs

Extract [`point_light.zip`](https://drive.google.com/file/d/1Wo-0iNRYZs02GAfwlZ9SsFdUWjze6IZ-/view?usp=sharing)  and move each scene to `./public_data/nerf_synthetic`. Then run

```bash
python exp_runner.py --mode train --conf ./confs/point_shadow.conf --case <case_name>_specular_point
```

#### On directional light RGB inputs

Extract [`directional_light.zip`](https://drive.google.com/file/d/10tla4ZygVIqOUUOUt_cBOVFLyxcRYH8z/view?usp=sharing) and move each scene to `./public_data/nerf_synthetic`. Then run

```bash
python exp_runner.py --mode train --conf ./confs/directional_color.conf --case <case_name>_specular
```

#### On directional light shadow inputs

Extract [`directional_light.zip`](https://drive.google.com/file/d/10tla4ZygVIqOUUOUt_cBOVFLyxcRYH8z/view?usp=sharing)  and move each scene to `./public_data/nerf_synthetic`. Then run

```bash
python exp_runner.py --mode train --conf ./confs/directional_shadow.conf --case <case_name>_specular
```

#### On vertical-down shadow inputs

Extract [`vertical_down.zip`](https://drive.google.com/file/d/1YllYfPrHsWA5zCcnXh-UpRpsA8NmVm8b/view?usp=sharing)  and move each scene to `./public_data/nerf_synthetic`. Then run

```bash
python exp_runner.py --mode train --conf ./confs/point_shadow.conf --case <case_name>_upup
```

#### On real data

Extract [`real_data.zip`](https://drive.google.com/file/d/1OJsumvYIPwB7AdtfR2CT5hPphrQGG2dM/view?usp=sharing)  to `./public_data` and run

```bash
python exp_runner.py --mode train --conf ./confs/real_data.conf --case <case_name>
```

#### On DeepShadow Dataset

You can download [`DeepShadowData.zip`](https://faculty.runi.ac.il/toky/Pub/DeepShadowData.zip) from their [project page](https://asafkar.github.io/deepshadow/), and unzip it to `./public_data`. Then run


```bash
python exp_runner.py --mode train --conf ./confs/deepshadow.conf --case <case_name>
```

## Citation

Cite as below if you find this repository helpful:

```
@misc{ling2022shadowneus,
    title={ShadowNeuS: Neural SDF Reconstruction by Shadow Ray Supervision}, 
    author={Jingwang Ling and Zhibo Wang and Feng Xu},
    year={2022},
    eprint={2211.14086},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgement

The project structure is based on [NeuS](https://github.com/Totoro97/NeuS). Some code is borrowed from [deep_shadow](https://github.com/asafkar/deep_shadow), [IRON](https://github.com/Kai-46/IRON) and [psnerf](https://github.com/ywq/psnerf). Thanks for these great projects.
