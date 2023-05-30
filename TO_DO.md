## To-Do

### Pre-Alpha Release 0.1.0 (Train Backbones with ImageNet-1k)

- [x] Argparse problem: initializing parse arguments in a python script, but passing those arguments from bash terminal while running another python script. The goal is to make training script dynamic for any model training.
- [ ] writing utils/imagenet_1k_dataset_script.py similar to coco_script availale online for working with local imagenet files for getting image as huggingface datasets format
- [ ] writing download script: imagenet_1k_download_and_extract.sh
- [ ] general config and preprocess_config backbones: bit
- [ ] general config and preprocess_config backbones: convnext
- [ ] general config and preprocess_config backbones: convnextv2
- [ ] general config and preprocess_config backbones: dinat
- [ ] general config and preprocess_config backbones: focalnet
- [ ] general config and preprocess_config backbones: maskformer-swin
- [ ] general config and preprocess_config backbones: nat
- [ ] general config and preprocess_config backbones: resnet
- [ ] general config and preprocess_config backbones: swin
- [ ] backbone model and preprocessor loading scripts: bit.py 
- [ ] backbone model and preprocessor loading scripts: convnext.py
- [ ] backbone model and preprocessor loading scripts: convnextv2.py
- [ ] backbone model and preprocessor loading scripts: dinat.py
- [ ] backbone model and preprocessor loading scripts: focalnet.py
- [ ] backbone model and preprocessor loading scripts: maskformer-swin.py
- [ ] backbone model and preprocessor loading scripts: nat.py 
- [ ] backbone model and preprocessor loading scripts: resnet.py 
- [ ] backbone model and preprocessor loading scripts: swin.py
- [ ] training_script_imagenet_1k.py (load config from dict path, save path (temporary))
- [ ] inference_script_imagenet_1k.ipynb
- [ ] training_imagenet_1k bash command (readme, trial)
- [ ] inference_imagenet_1k (readme, trial)


### Future Releases

- [x] writing download script: coco_datasets_download_and_extract.sh
- [ ] general config and preprocess_config architectures: DeTR
- [ ] general config and preprocess_config architectures: maskformer
- [ ] general config and preprocess_config architectures: mask2former
- [ ] general config and preprocess_config architectures: oneformer
- [ ] architecture model and preprocessor loading scripts: DeTR.py
- [ ] architecture model and preprocessor loading scripts: maskformer.py
- [ ] architecture model and preprocessor loading scripts: mask2former.py
- [ ] architecture model and preprocessor loading scripts: oneformer.py
- [ ] download_script: coco_download_and_extract.sh
- [ ] training_script_coco_panoptic.py (load config from dict path, pretrained_weight_backbone_load_path, save_path (temporary))
- [ ] inference_script_coco_panoptic.ipynb
- [ ] training_coco_panoptic bash command (readme, trial)
- [ ] inference_coco_panoptic (readme, trial)

