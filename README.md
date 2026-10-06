# Лабораторная работа: векторизация текста и классификация тональности

Выполнены все пять заданий: подготовка SST-2, TF-IDF, N-граммы, усреднённые GloVe-эмбеддинги, признаки DistilBERT и сравнение результатов с ответами на контрольные вопросы.

Основной файл — **[lab.ipynb](lab.ipynb)**. В нём находятся пояснения, код и сохранённые результаты выполнения. Для просмотра и повторного запуска откройте его в VS Code с расширениями Python и Jupyter, затем выберите ядро **`venv/bin/python`** из этой папки и выполните ячейки по порядку (`Run All`).

## Запуск из терминала

В текущей среде библиотеки уже установлены. Из папки проекта запустите:

```bash
./venv/bin/python lab.py
```

Скрипт выполняет все эксперименты и сохраняет таблицу метрик в `results/metrics.csv`, подробные результаты в `results/metrics.json`, итоговый анализ и ответы — в `results/report.md`.

Параметры для запуска на CPU и принудительного пересчёта признаков DistilBERT:

```bash
./venv/bin/python lab.py --batch-size 16 --threads 2 --device cpu --refresh-features
```

На первом запуске нужен интернет: скачиваются данные, GloVe и DistilBERT, суммарно около 400 МБ. Расчёт на CPU может занять несколько минут. Повторные запуски используют локальный кэш: `cache/gensim/`, `cache/huggingface/` и `cache/distilbert_features.npz`. Датасет сохраняется в `data/sst2_train.tsv`.

## Что сравнивается

| Метод | Представление текста | Классификатор |
| --- | --- | --- |
| TF-IDF (Unigrams) | `ngram_range=(1, 1)`, `min_df=5` | LogisticRegression |
| TF-IDF (N-Grams) | `ngram_range=(1, 2)`, `min_df=5` | LogisticRegression |
| GloVe (Averaged) | Средний вектор известных слов, `glove-twitter-25`, 25 признаков | LogisticRegression |
| DistilBERT | Последний скрытый слой токена `[CLS]`, `distilbert-base-uncased`, 768 признаков | LogisticRegression |

Используются первые **2000 строк** SST-2. Для всех методов применяется одно разбиение: **1500 обучающих и 500 тестовых примеров**, `random_state=42`, без стратификации. TF-IDF обучается только на тренировочных текстах, затем преобразует тестовые. Предобученные GloVe и DistilBERT используются для извлечения признаков без дообучения. Итоговая таблица содержит фактически рассчитанные `train accuracy` и `test accuracy`.

## Установка в новой среде

Этот шаг нужен только при переносе проекта или создании новой среды. Рекомендуется Python 3.12:

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python lab.py
```

Версии библиотек среды, в которой выполнена работа, записаны в `environment.txt`. Небольшие различия метрик при запуске с другими версиями библиотек или на другом устройстве возможны.

Источники: [SST-2 из условия задания](https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv), [документация DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert), [Gensim Downloader](https://radimrehurek.com/gensim/downloader.html).
