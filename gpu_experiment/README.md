# GPU experiment

Запуск из корня репозитория:

```bash
python gpu_experiment/amc_cnn_experiment.py --output-dir results
```

Установка зависимостей:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r gpu_experiment/requirements.txt
```

Проверка CPU-режима:

```bash
python gpu_experiment/amc_cnn_experiment.py --allow-cpu --epochs 1 --train-per-class-snr 2 --test-per-class-snr 1 --output-dir temp/smoke_results
```

Аргументы:

```bash
python gpu_experiment/amc_cnn_experiment.py --help
```
