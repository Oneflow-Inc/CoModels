## Poolformer-M36

### Installation
- Install the latest version of OneFlow
```bash
python3 -m pip install oneflow -f https://staging.oneflow.info/branch/master/[PLATFORM]
```
Find more information on [install oneflow](https://github.com/Oneflow-Inc/oneflow#install-oneflow)

- Install flowvision

Then install the latest stable release of flowvision

```bash
pip install flowvision==0.2.1
```

- Install other requirements
```bash
python3 -m pip install -r requirements.txt
```


### Dataset
#### ImageNet
For ImageNet dataset, you can download it from http://image-net.org/.

### Code Structure


  ```
    .
    ├── configs -> ../configs
    │   ├── ...
    │   └── default_settings.yaml
    ├── data -> ../data
    │   ├── __init__.py
    │   ├── build.py
    │   ├── cached_image_folder.py
    │   ├── samplers.py
    │   └── zipreader.py
    ├── utils.py -> ../utils.py
    ├── config.py -> ../config.py
    ├── logger.py -> ../logger.py
    ├── lr_scheduler.py -> ../lr_scheduler.py
    ├── optimizer.py -> ../optimizer.py
    ├── main.py
    ├── train.sh
    └── infer.sh
  ```



### Training
You can use bash script `train.sh` to train this model.
```````
sh train.sh
```````

### Inference

Bash script `infer.sh` is used to infer the trained model.
```````
sh infer.sh
```````
