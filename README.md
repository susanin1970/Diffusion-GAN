## Diffusion-GAN &mdash; официальная реализация на PyTorch

![Иллюстрация](./docs/diffusion-gan.png)

**Diffusion-GAN: Training GAN's with Diffusion**<br>
Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen и Mingyuan Zhou <br>
https://arxiv.org/abs/2206.02262 <br>

Аннотация: *Для стабильного обучения генеративно-состязательных сетей (GAN) добавление случайного шума к входу дискриминатора считается теоретически обоснованным решением, однако на практике его потенциал пока не удалось полностью реализовать. В этой работе представлен Diffusion-GAN: для добавления шума к экземплярам данных используется смесь гауссовских распределений, заданная на всех шагах прямой цепочки диффузии. На каждом шаге случайно выбирается уровень шума, и зашумлённые реальные или сгенерированные данные подаются на вход дискриминатору. Генератор обновляется с помощью обратного распространения градиента через прямую цепочку диффузии. Её длина адаптивно меняется, чтобы ограничивать максимально допустимое отношение шума к сигналу на каждом шаге обучения. Теоретический анализ подтверждает обоснованность предложенного метода. Diffusion-GAN обеспечивает дифференцируемую аугментацию, не зависящую от модели и предметной области. Масштабные эксперименты на различных наборах данных показывают, что метод позволяет стабильно обучать GAN с эффективным использованием данных и последовательно превосходит сильные базовые модели при генерации фотореалистичных изображений.*

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusion-gan-training-gans-with-diffusion/image-generation-on-celeba-64x64)](https://paperswithcode.com/sota/image-generation-on-celeba-64x64?p=diffusion-gan-training-gans-with-diffusion)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusion-gan-training-gans-with-diffusion/image-generation-on-stl-10)](https://paperswithcode.com/sota/image-generation-on-stl-10?p=diffusion-gan-training-gans-with-diffusion)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusion-gan-training-gans-with-diffusion/image-generation-on-lsun-bedroom-256-x-256)](https://paperswithcode.com/sota/image-generation-on-lsun-bedroom-256-x-256?p=diffusion-gan-training-gans-with-diffusion)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusion-gan-training-gans-with-diffusion/image-generation-on-afhq-wild)](https://paperswithcode.com/sota/image-generation-on-afhq-wild?p=diffusion-gan-training-gans-with-diffusion)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusion-gan-training-gans-with-diffusion/image-generation-on-afhq-cat)](https://paperswithcode.com/sota/image-generation-on-afhq-cat?p=diffusion-gan-training-gans-with-diffusion)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusion-gan-training-gans-with-diffusion/image-generation-on-afhq-dog)](https://paperswithcode.com/sota/image-generation-on-afhq-dog?p=diffusion-gan-training-gans-with-diffusion)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusion-gan-training-gans-with-diffusion/image-generation-on-lsun-churches-256-x-256)](https://paperswithcode.com/sota/image-generation-on-lsun-churches-256-x-256?p=diffusion-gan-training-gans-with-diffusion)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusion-gan-training-gans-with-diffusion/image-generation-on-ffhq-1024-x-1024)](https://paperswithcode.com/sota/image-generation-on-ffhq-1024-x-1024?p=diffusion-gan-training-gans-with-diffusion)

## Что сделано
- [x] Опубликован исходный код
- [x] Добавлены предобученные модели

## Как создать Diffusion-GAN
В этом разделе описано, как обучать GAN с помощью диффузии. Мы предлагаем два подхода:
1. Подключить диффузию как обычный метод аугментации данных.
2. Обучать GAN на цепочках диффузии, используя дискриминатор, учитывающий шаг диффузии.
Пока мы не обнаружили существенных экспериментальных различий между этими подходами, хотя второй имеет более строгие теоретические обоснования. Мы предполагаем, что второй подход может дать лучшие результаты при использовании в дискриминаторе более сложной структуры, зависящей от шага диффузии. Это направление оставлено для дальнейшего исследования.

### Простой способ подключения
* Реализуйте подходящий процесс диффузии на основе файла ```diffusion.py``` ([пример](https://github.com/Zhendong-Wang/Diffusion-GAN/blob/main/diffusion-stylegan2/training/diffusion.py)).
* Применяйте диффузию к входным данным дискриминаторов:
```logits = Discriminator(Diffusion(gen/real_images))```
* Добавьте адаптивную настройку диффузии в цикл обучения:
``` 
if update_diffusion:  # batch_idx % ada_interval == 0
    adjust = np.sign(sign(Discriminator(real_images)) - ada_target) * C  # C = (batch_size * ada_interval) / (ada_kimg * 1000)
    diffusion.p = (diffusion.p + adjust).clip(min=0., max=1.)
    diffusion.update_T()
```

### Полная версия
* Передавайте шаг диффузии `t` на вход дискриминатора: `logits = Discriminator(images, t)`.
  Для этого может потребоваться изменить архитектуру дискриминатора.
* Остальные шаги такие же, как в простом способе подключения. Поскольку дискриминатор зависит от шага диффузии, необходимо также передавать значение `t`.
```
diffused_images, t = Diffusion(images)
logits = Discriminator(diffused_images, t)
```

## Обучение Diffusion-GAN

### Требования
В каждой папке находится файл `environment.yml` для подготовки окружения. Для совместимости с более новыми версиями PyTorch в каталоге `torch_utils` были внесены изменения, поэтому результаты могут немного различаться в зависимости от версии PyTorch. Предоставленные контрольные точки были получены в старом окружении, унаследованном от [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch#requirements). Его основные требования:
* 64-разрядная версия Python 3.7 и PyTorch 1.7.1 или 1.8.1. Инструкции по установке PyTorch см. на [pytorch.org](https://pytorch.org/).
* CUDA Toolkit версии 11.0 или новее.
* Библиотеки Python: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`.

### Подготовка данных

В работе модели обучались на наборах данных [CIFAR-10 (32 x 32)](https://www.cs.toronto.edu/~kriz/cifar.html), [STL-10 (64 x 64)](https://cs.stanford.edu/~acoates/stl10/),
[LSUN (256 x 256)](https://github.com/fyu/lsun), [AFHQ (512 x 512)](https://github.com/clovaai/stargan-v2) и [FFHQ (1024 x 1024)](https://github.com/NVlabs/ffhq-dataset).
Эти наборы можно скачать с соответствующих сайтов. Чтобы подготовить данные с нужным разрешением, выполните, например, следующие команды:
```.bash
python dataset_tool.py --source=~/downloads/lsun/raw/bedroom_lmdb --dest=~/datasets/lsun_bedroom200k.zip \
    --transform=center-crop --width=256 --height=256 --max_images=200000

python dataset_tool.py --source=~/downloads/lsun/raw/church_lmdb --dest=~/datasets/lsun_church200k.zip \
    --transform=center-crop-wide --width=256 --height=256 --max_images=200000
```

### Обучение

Ниже приведены команды, использованные для обучения. Команды для разных наборов данных в основном похожи; в качестве примера показано обучение на CIFAR-10.

Для Diffusion-GAN:
```.bash
python train.py --outdir=training-runs --data="~/cifar10.zip" --gpus=4 --cfg cifar --kimg 50000 --aug no --target 0.6 --noise_sd 0.05 --ts_dist priority
```
Для Diffusion-ProjectedGAN:
```.bash
python train.py --outdir=training-runs --data="~/cifar10.zip" --gpus=4 --batch 64 --batch-gpu=16 --cfg fastgan --kimg 50000 --target 0.45 --d_pos first --noise_sd 0.5
```
Для Diffusion-InsGen:
```.bash
python train.py --outdir=training-runs --data="~/afhq-wild.zip" --gpus=8 --cfg paper512 --kimg 25000
```

Настройки `config` основаны на конфигурациях [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorchhttps://github.com/NVlabs/stylegan2-ada-pytorch); подробности см. в его документации. Основные гиперпараметры:
* `--target` — целевой показатель дискриминатора, регулирующий интенсивность диффузии.
* `--aug` — доменно-специфичная аугментация изображений, например ADA или Differentiable Augmentation. Она используется, чтобы оценить, насколько такие аугментации дополняют диффузию.
* `--noise_sd` — стандартное отклонение шума диффузии; в наших экспериментах оно равно 0.05.
* `--ts_dist` — распределение выборки шага `t`, обозначенное в статье как $\pi(t)$.

Мы сравнили два распределения для выборки `t`: `['priority', 'uniform']`. Вариант `'priority'` соответствует формуле (11) из статьи, а `'uniform'` означает равномерную случайную выборку. В большинстве случаев `'priority'` показывает немного лучшие результаты, однако для некоторых наборов, например FFHQ, лучше работает `'uniform'`.

## Генерация и оценка с помощью контрольных точек
Контрольные точки Diffusion-GAN доступны на странице проекта на [Hugging Face](https://huggingface.co/zhendongw/diffusion-gan/tree/main/checkpoints). Для удобства ссылки на них также приведены ниже.

|            Модель            |   Набор данных   | Разрешение |  FID  |                                                        Файл модели                                                         |
|:---------------------------:|:------------:|:----------:|:-----:|:-------------------------------------------------------------------------------------------------------------------------:|
|     Diffusion-StyleGAN2     |   CIFAR-10   |   32x32    | 3.19  |     [скачать](https://huggingface.co/zhendongw/diffusion-gan/resolve/main/checkpoints/diffusion-stylegan2-cifar10.pkl)     |
|     Diffusion-StyleGAN2     |    CelebA    |   64x64    | 1.69  |    [скачать](https://huggingface.co/zhendongw/diffusion-gan/resolve/main/checkpoints/diffusion-stylegan2-celeba64.pkl)    |
|     Diffusion-StyleGAN2     |    STL-10    |   64x64    | 11.53 |      [скачать](https://huggingface.co/zhendongw/diffusion-gan/resolve/main/checkpoints/diffusion-stylegan2-stl10.pkl)      |
|     Diffusion-StyleGAN2     | LSUN-Bedroom |  256x256   | 3.65  |  [скачать](https://huggingface.co/zhendongw/diffusion-gan/resolve/main/checkpoints/diffusion-stylegan2-lsun-bedroom.pkl)   |
|     Diffusion-StyleGAN2     | LSUN-Church  |  256x256   | 3.17  |   [скачать](https://huggingface.co/zhendongw/diffusion-gan/resolve/main/checkpoints/diffusion-stylegan2-lsun-church.pkl)   |
|     Diffusion-StyleGAN2     |     FFHQ     | 1024x1024  | 2.83  |      [скачать](https://huggingface.co/zhendongw/diffusion-gan/resolve/main/checkpoints/diffusion-stylegan2-ffhq.pkl)       |
|   Diffusion-ProjectedGAN    |   CIFAR-10   |   32x32    | 2.54  |   [скачать](https://huggingface.co/zhendongw/diffusion-gan/resolve/main/checkpoints/diffusion-projectedgan-cifar10.pkl)    |
|   Diffusion-ProjectedGAN    |    STL-10    |   64x64    | 6.91  |    [скачать](https://huggingface.co/zhendongw/diffusion-gan/resolve/main/checkpoints/diffusion-projectedgan-stl10.pkl)     |
|   Diffusion-ProjectedGAN    | LSUN-Bedroom |  256x256   | 1.43  | [скачать](https://huggingface.co/zhendongw/diffusion-gan/resolve/main/checkpoints/diffusion-projectedgan-lsun-bedroom.pkl) |
|   Diffusion-ProjectedGAN    | LSUN-Church  |  256x256   | 1.85  | [скачать](https://huggingface.co/zhendongw/diffusion-gan/resolve/main/checkpoints/diffusion-projectedgan-lsun-church.pkl)  |
|      Diffusion-InsGen       |   AFHQ-Cat   |  512x512   | 2.40  |      [скачать](https://huggingface.co/zhendongw/diffusion-gan/resolve/main/checkpoints/diffusion-insgen-afhqcat.pkl)      |
|     Diffusion-InsGen        |   AFHQ-Dog   |  512x512   | 4.83  |      [скачать](https://huggingface.co/zhendongw/diffusion-gan/resolve/main/checkpoints/diffusion-insgen-afhqdog.pkl)      |
|      Diffusion-InsGen       |   AFHQ-Wild  |  512x512    | 1.51  | [скачать](https://huggingface.co/zhendongw/diffusion-gan/resolve/main/checkpoints/diffusion-insgen-afhqwild.pkl)      |


Чтобы сгенерировать изображения, выполните следующие команды:

```.bash
# Генерация изображений FFHQ с помощью предобученной Diffusion-StyleGAN2
python generate.py --outdir=out --seeds=1-100 \
    --network=https://tsciencescu.blob.core.windows.net/projectshzheng/DiffusionGAN/diffusion-stylegan2-ffhq.pkl

# Генерация изображений LSUN-Church с помощью предобученной Diffusion-ProjectedGAN
python gen_images.py --outdir=out --seeds=1-100 \
    --network=https://tsciencescu.blob.core.windows.net/projectshzheng/DiffusionGAN/diffusion-projectedgan-lsun-church.pkl
```

Вместо указанных контрольных точек можно использовать любой файл предобученной модели Diffusion-GAN, скачанный по ссылкам из таблицы выше.


Метрики можно вычислить аналогичным образом:

```.bash
# Файл предобученной модели: укажите набор данных явно; результат будет выведен в stdout.
python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq.zip --mirror=1 \
    --network=https://tsciencescu.blob.core.windows.net/projectshzheng/DiffusionGAN/diffusion-stylegan2-ffhq.pkl
```

## Цитирование

```
@article{wang2022diffusiongan,
  title     = {Diffusion-GAN: Training GANs with Diffusion},
  author    = {Wang, Zhendong and Zheng, Huangjie and He, Pengcheng and Chen, Weizhu and Zhou, Mingyuan},
  journal   = {arXiv preprint arXiv:2206.02262},
  year      = {2022},
  url       = {https://arxiv.org/abs/2206.02262}
}
```

## Отличия данного репозитория от оригинала

В рамках работы над диссертационным исследованием была проведена замена процедуры прямоу диффузии у варианта Diffusion-ProjectedGAN на процедуру, которая использует косинусоидальное расписание шума: (см. diffusion-projected-gan\pg_modules\diffusion.py)

## Благодарности

В основе нашего кода лежат проекты [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), [InsGen](https://github.com/genforce/insgen) и [ProjectedGAN](https://github.com/autonomousvision/projected_gan), созданные соответственно Karras и соавторами, Ceyuan Yang и соавторами, а также Axel Sauer и соавторами.
