# TianPeng

Fine-tuning and Inference Pipeline based on LLaMA, and Multi-modal Chatbot implementation based on it.

## Development

Instructions on how to set up the environment for development are available [here](./docs/SETUP_DEV_ENV.md).

### Quickstart

```bash

# with  pyenv
pyenv shell 3.10

# Install dependencies
yarn install
pdm install # build and install python dependencies

## if conda is to be used
conda create -n tp python=3.10
conda activate tp
conda deactivate

## Run the app
cd tianpeng/app
python main.app

## Commit Changes
yarn commit # use `yarn commit` instead of `git add . && git commit` to commit changes
```

### FineTuning

```bash
pip install flash-attn
pdm run torchrun --nproc_per_node=8 tianpeng/finetune/main.py --model_config_file train_config/Llama_config.json --lora_hyperparams_file train_config/lora_hyperparams_llama.json  --use_lora
```

### Citation

```bibtex
@misc{tianpeng,
  author = {Li, Ding and Xian, Zhang},
  title = {TianPeng: A Fine-tuning and Inference Pipeline based on LLaMA},
  howpublished = {\url{https://github.com/pleisto/tianpeng}},
  year = {2023}
}
```

## License

TianPeng is licensed under the [GNU General Public License v3.0](./LICENSE).
