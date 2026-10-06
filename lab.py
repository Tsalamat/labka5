#!/usr/bin/env python3
"""Все пять частей лабораторной: SST-2, TF-IDF, GloVe и DistilBERT.

Запуск: ``venv/bin/python lab.py``. При первом запуске нужны интернет и место
для исходных данных, GloVe и DistilBERT. Повторные запуски используют кэш.
Функции этого файла также используются в учебном Jupyter Notebook.
"""

from __future__ import annotations

import os
from pathlib import Path

# Задаём доступные для записи каталоги ДО импортов gensim/transformers.
PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "cache"
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
os.environ.setdefault("GENSIM_DATA_DIR", str(CACHE_DIR / "gensim"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import re
import time
from datetime import datetime, timezone
from urllib.request import urlopen
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DATA_URL = (
    "https://github.com/clairett/pytorch-sentiment-classification/"
    "raw/master/data/SST2/train.tsv"
)
DATA_PATH = DATA_DIR / "sst2_train.tsv"
MODEL_NAME = "distilbert-base-uncased"
GLOVE_NAME = "glove-twitter-25"
FEATURES_PATH = CACHE_DIR / "distilbert_features.npz"
RANDOM_STATE = 42
N_SAMPLES = 2000
MAX_LENGTH = 512
TOKEN_PATTERN = re.compile(r"\b\w+(?:'\w+)?\b|[^\w\s]", flags=re.UNICODE)


def progress(message: str) -> None:
    """Печать прогресса без буферизации, в том числе при запуске из терминала."""
    print(message, flush=True)


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Загрузить весь TSV SST-2; 2000 первых строк выбираются вызывающим кодом.

    Файл не имеет заголовка: колонка 0 — предложение, колонка 1 — метка 0/1.
    Уже сохранённый файл повторно не скачивается.
    """
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        progress(f"Скачивание SST-2 → {path}")
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with urlopen(DATA_URL, timeout=120) as response:
                contents = response.read()
            temporary.write_bytes(contents)
            temporary.replace(path)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Не удалось скачать SST-2. Проверьте интернет или сохраните "
                f"исходный train.tsv в {path}. Причина: {exc}"
            ) from exc
    df = pd.read_csv(path, sep="\t", header=None, keep_default_na=False)
    if df.shape[1] != 2 or len(df) < N_SAMPLES:
        raise ValueError(f"Ожидались минимум {N_SAMPLES} строк и две колонки: {path}")
    df[0] = df[0].astype(str)
    labels = pd.to_numeric(df[1], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("В SST-2 ожидаются только метки 0 и 1.")
    df[1] = labels.astype(np.int64)
    if df[0].str.strip().eq("").any():
        raise ValueError("В исходном датасете обнаружены пустые тексты.")
    return df


def split_indices(batch_1: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Одно разбиение 75/25 для ВСЕХ подходов, как в условии (без stratify).

    Возвращает позиционные индексы для ``batch_1.iloc[indices]``.
    """
    return train_test_split(
        np.arange(len(batch_1)), test_size=0.25, random_state=RANDOM_STATE
    )


def evaluate_features(
    name: str,
    train_features,
    test_features,
    y_train,
    y_test,
) -> tuple[LogisticRegression, dict]:
    """Обучить одинаковую LR и вернуть классификатор и реальные метрики."""
    started = time.perf_counter()
    classifier = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    classifier.fit(train_features, y_train)
    train_accuracy = float(accuracy_score(y_train, classifier.predict(train_features)))
    test_accuracy = float(accuracy_score(y_test, classifier.predict(test_features)))
    record = {
        "method": name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "gap": train_accuracy - test_accuracy,
        "n_features": int(train_features.shape[1]),
        "classifier_seconds": time.perf_counter() - started,
    }
    progress(
        f"{name}: train={train_accuracy:.4f}; test={test_accuracy:.4f}; "
        f"признаков={record['n_features']}"
    )
    return classifier, record


def load_glove():
    """Загрузить настоящие предобученные векторы GloVe (25 измерений)."""
    import gensim.downloader

    Path(os.environ["GENSIM_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    progress("Загрузка glove-twitter-25 (при первом запуске потребуется скачивание)…")
    return gensim.downloader.load("glove-twitter-25")


def get_sentence_embedding(sentence: str, glove_vectors) -> np.ndarray:
    """Усреднить векторы известных токенов; если все OOV — вернуть нули.

    Регистр приводится к нижнему. Слова, числа и пунктуация выделяются регулярным
    выражением. Неизвестные токены не входят в знаменатель среднего.
    """
    tokens = TOKEN_PATTERN.findall(str(sentence).lower())
    vectors = [glove_vectors[token] for token in tokens if token in glove_vectors]
    if not vectors:
        return np.zeros(glove_vectors.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0, dtype=np.float32)


def load_distilbert(device: str = "cpu"):
    """Вернуть (DistilBertTokenizer, DistilBertModel) в режиме eval()."""
    from transformers import DistilBertModel, DistilBertTokenizer

    progress(f"Загрузка {MODEL_NAME} на {device}…")
    # Сначала проверяем локальный кэш, чтобы повторный запуск работал без сети
    # и не ожидал HTTP HEAD к Hugging Face.
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    except OSError:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    try:
        model = DistilBertModel.from_pretrained(MODEL_NAME, local_files_only=True)
    except OSError:
        model = DistilBertModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model


def prepare_bert_inputs(texts, tokenizer) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    """Ручные encode → padding нулями → attention_mask из задания.

    Последовательности длиннее 512 токенов усекаются с сохранением специальных
    токенов. Максимальная длина определяется по всем переданным предложениям.
    """
    texts = list(texts)
    if not texts:
        raise ValueError("Нельзя токенизировать пустой список текстов.")
    if tokenizer.pad_token_id != 0:
        raise ValueError("В этом задании нужен DistilBERT с pad_token_id=0.")
    tokenized = [
        tokenizer.encode(
            str(text), add_special_tokens=True, truncation=True, max_length=MAX_LENGTH
        )
        for text in texts
    ]
    max_len = max(len(tokens) for tokens in tokenized)
    padded = np.zeros((len(tokenized), max_len), dtype=np.int64)
    for i, tokens in enumerate(tokenized):
        padded[i, : len(tokens)] = tokens
    attention_mask = (padded != 0).astype(np.int64)
    return tokenized, padded, attention_mask


def _texts_fingerprint(texts: list[str]) -> str:
    payload = json.dumps(texts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_bert_features(
    texts,
    tokenizer=None,
    model=None,
    *,
    batch_size: int = 16,
    device: str = "cpu",
    refresh: bool = False,
    threads: int = 2,
    cache_path: str | Path = FEATURES_PATH,
) -> np.ndarray:
    """Извлечь CLS (n, 768) небольшими батчами и сохранить проверяемый кэш.

    При совпадении текстов, их порядка и имени модели возвращается кэш. Если
    tokenizer/model не переданы, они загружаются только при отсутствии кэша.
    Переданные объекты должны быть исходными distilbert-base-uncased: кэш не
    предназначен для пользовательских дообученных весов.
    """
    if batch_size < 1 or threads < 1:
        raise ValueError("batch_size и threads должны быть положительными.")
    if (tokenizer is None) != (model is None):
        raise ValueError("Передайте одновременно tokenizer и model либо не передавайте оба.")
    if model is not None:
        model_name = str(getattr(model.config, "_name_or_path", ""))
        tokenizer_name = str(getattr(tokenizer, "name_or_path", ""))
        if model_name != MODEL_NAME or tokenizer_name != MODEL_NAME:
            raise ValueError(
                f"Кэш рассчитан на исходные tokenizer/model {MODEL_NAME}; "
                "загрузите их через load_distilbert()."
            )
    texts = [str(text) for text in texts]
    if not texts:
        raise ValueError("Нет текстов для извлечения признаков.")
    cache_path = Path(cache_path)
    expected_metadata = {
        "format_version": 1,
        "model": MODEL_NAME,
        "text_sha256": _texts_fingerprint(texts),
        "max_length": MAX_LENGTH,
        "pooling": "last_hidden_state[:, 0, :]",
    }
    if cache_path.exists() and not refresh:
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                metadata = json.loads(str(cached["metadata"].item()))
                features = cached["features"]
                if (
                    metadata == expected_metadata
                    and features.shape == (len(texts), 768)
                    and np.issubdtype(features.dtype, np.floating)
                    and np.isfinite(features).all()
                ):
                    progress(f"DistilBERT: проверенный кэш признаков {features.shape}")
                    return features.astype(np.float32, copy=False)
            progress("Кэш DistilBERT не соответствует текущим данным; пересчитываем.")
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            progress(f"Кэш DistilBERT повреждён ({exc}); пересчитываем.")
    torch.set_num_threads(threads)
    torch.manual_seed(RANDOM_STATE)
    if tokenizer is None:
        tokenizer, model = load_distilbert(device=device)
    if (
        getattr(model.config, "model_type", None) != "distilbert"
        or getattr(model.config, "dim", None) != 768
    ):
        raise ValueError("Для этого задания нужен DistilBertModel с размерностью 768.")
    model.to(device)
    model.eval()
    _, padded, attention_mask = prepare_bert_inputs(texts, tokenizer)
    progress(
        f"DistilBERT: padded={padded.shape}, batch_size={batch_size}, "
        f"threads={threads}; извлекаем CLS…"
    )
    features = np.empty((len(texts), 768), dtype=np.float32)
    started = time.perf_counter()
    last_update = started
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            stop = min(start + batch_size, len(texts))
            # Убираем только правый padding, общий для всего мини-батча.
            # Значимые токены и маски остаются теми же, вычислений требуется меньше.
            batch_max_len = int(attention_mask[start:stop].sum(axis=1).max())
            input_ids = torch.tensor(
                padded[start:stop, :batch_max_len], dtype=torch.long, device=device
            )
            mask = torch.tensor(
                attention_mask[start:stop, :batch_max_len], dtype=torch.long, device=device
            )
            last_hidden_states = model(input_ids=input_ids, attention_mask=mask)
            features[start:stop] = last_hidden_states[0][:, 0, :].cpu().numpy()
            now = time.perf_counter()
            if stop == len(texts) or start == 0 or now - last_update >= 15:
                progress(f"  CLS: {stop}/{len(texts)} текстов; прошло {now - started:.1f} с")
                last_update = now
    if not np.isfinite(features).all():
        raise ValueError("DistilBERT вернул нечисловые или бесконечные признаки.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".part.npz")
    np.savez_compressed(
        temporary,
        features=features,
        metadata=np.array(json.dumps(expected_metadata, ensure_ascii=False)),
    )
    temporary.replace(cache_path)
    return features


def save_results(
    records: list[dict],
    batch_1: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: str | Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Сохранить метрики CSV/JSON и русский отчёт с ответами на все вопросы."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(records)
    results.to_csv(output_dir / "metrics.csv", index=False)
    y_train = batch_1.iloc[train_idx][1].to_numpy()
    y_test = batch_1.iloc[test_idx][1].to_numpy()
    majority_class = int(pd.Series(y_train).mode().iloc[0])
    baseline_train = float(np.mean(y_train == majority_class))
    baseline_test = float(np.mean(y_test == majority_class))
    counts = {str(k): int(v) for k, v in batch_1[1].value_counts().sort_index().items()}
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_url": DATA_URL,
        "n_samples": len(batch_1),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "random_state": RANDOM_STATE,
        "stratify": None,
        "package_versions": {
            package: version(package)
            for package in ["numpy", "pandas", "torch", "transformers", "scikit-learn", "gensim", "scipy"]
        },
        "class_counts": counts,
        "majority_baseline": {
            "class_chosen_from_train": majority_class,
            "train_accuracy": baseline_train,
            "test_accuracy": baseline_test,
        },
        "models": records,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best_score = float(results["test_accuracy"].max())
    winners = results.loc[np.isclose(results["test_accuracy"], best_score), "method"].tolist()
    rows = "\n".join(
        f"| {r['method']} | {r['train_accuracy']:.4f} | {r['test_accuracy']:.4f} | "
        f"{r['gap']:+.4f} | {int(r['n_features'])} |"
        for r in records
    )
    winner_explanations = []
    if any("TF-IDF" in name for name in winners):
        winner_explanations.append(
            "TF-IDF хорошо выделяет характерные для тональности слова; "
            "вариант с биграммами дополнительно учитывает локальные сочетания слов."
        )
    if any("GloVe" in name for name in winners):
        winner_explanations.append(
            "Усреднённые GloVe дают компактное представление семантики; "
            "небольшая размерность ограничивает сложность классификатора на малой выборке."
        )
    if any("DistilBERT" in name for name in winners):
        winner_explanations.append(
            "DistilBERT использует контекст и знания, полученные при предобучении "
            "на большом корпусе, поэтому может распознавать более сложные сочетания слов."
        )
    comparison = []
    unigram = results[results["method"] == "TF-IDF (Unigrams)"]
    ngram = results[results["method"] == "TF-IDF (N-Grams)"]
    if len(unigram) and len(ngram):
        delta = float(ngram.iloc[0]["test_accuracy"] - unigram.iloc[0]["test_accuracy"])
        comparison.append(
            f"Переход от униграмм к униграммам и биграммам изменил test accuracy "
            f"на {delta * 100:+.2f} процентного пункта."
        )
    max_gap = results.loc[results["gap"].idxmax()]
    comparison.append(
        f"Наибольший разрыв train − test у {max_gap['method']}: "
        f"{max_gap['gap']:.4f}. Положительный разрыв указывает на возможное "
        f"переобучение, но одной величины разрыва недостаточно для строгого вывода."
    )
    report = f"""# Сравнительный анализ векторизации текста на SST-2

## 1. Данные и условия эксперимента

Использованы первые {len(batch_1)} строк [SST-2]({DATA_URL}).
Классы: 0 — негативный ({counts.get('0', 0)}), 1 — позитивный ({counts.get('1', 0)}).
Разбиение: {len(train_idx)} обучающих и {len(test_idx)} тестовых примеров,
`random_state=42`, без `stratify`. Позиционные индексы разбиения одинаковы у всех методов.
Словарь и IDF вычислены только по обучающей части. Предобученные GloVe и DistilBERT
не дообучались и не используют метки SST-2 при извлечении признаков.
Для каждого представления обучена `LogisticRegression(max_iter=2000, random_state=42)`.

Базовый классификатор всегда предсказывает наиболее частый **в обучающей части**
класс {majority_class}: train accuracy = {baseline_train:.4f}, test accuracy = {baseline_test:.4f}.

## 2–4. Представления текста

- **TF-IDF (Unigrams):** `min_df=5`, `ngram_range=(1, 1)`.
- **TF-IDF (N-Grams):** `min_df=5`, `ngram_range=(1, 2)`.
- **GloVe (Averaged):** `glove-twitter-25`, усреднение известных токенов после
  приведения к нижнему регистру и разделения регулярным выражением. Если известных
  токенов нет, используется нулевой вектор длины 25. Порядок слов не сохраняется.
- **DistilBERT (Embeddings + LR):** `distilbert-base-uncased`, ручные токенизация,
  padding и attention mask; 768 признаков из последнего скрытого слоя для `[CLS]`.
  Используются `eval()` и `torch.no_grad()`. Слишком длинные тексты усекаются до
  512 токенов; правый padding учитывается маской. Обработка небольшими батчами
  ограничивает расход памяти. Базовая модель специально не обучалась выдавать
  универсальные sentence embeddings; качество её `[CLS]` нельзя приравнивать к
  качеству специально обученных моделей Sentence Transformers или fine-tuning.

## 5. Реальные результаты запуска

| Подход / Модель | Train Accuracy | Test Accuracy | Train − Test | Признаков |
| --- | ---: | ---: | ---: | ---: |
{rows}

{' '.join(comparison)}

## Ответы на контрольные вопросы

**1. Какой метод показал наибольшую точность и почему?**

На этом разбиении лучший test accuracy равен **{best_score:.4f}**:
**{', '.join(winners)}**. {' '.join(winner_explanations)}
Это возможные объяснения результата, а не установленная причинная связь.
Разница лучшего результата с majority baseline равна
{(best_score - baseline_test) * 100:+.2f} процентного пункта.
На {len(test_idx)} тестовых объектах один правильный ответ меняет accuracy на
{100 / len(test_idx):.2f} процентного пункта; один запуск не доказывает превосходство
метода на любых данных. Для устойчивого вывода нужны другие разбиения или кросс-валидация.

**2. Чем контекстуализированные эмбеддинги отличаются от статических?**

В GloVe и Word2Vec каждому слову соответствует один вектор независимо от предложения.
У BERT/DistilBERT представление токена зависит от окружающих токенов и позиции:
одно и то же слово в разных контекстах получает разные векторы. Усреднение GloVe
теряет порядок слов; self-attention с позиционными эмбеддингами учитывает контекст.

**3. Для чего нужен `attention_mask`?**

После дополнения предложений до общей длины padding не должен влиять на внимание.
Маска содержит 1 для настоящих и специальных токенов и 0 для padding. Модель
не использует позиции с маской 0 как источники информации при self-attention.
Маска не удаляет padding из массива и не гарантирует нулевые выходные векторы в
позициях padding; в этом задании берётся только вектор первого токена.

**4. Почему выбирается `[:, 0, :]`?**

Последний скрытый слой имеет форму `(число текстов, число токенов, 768)`.
Первая ось `:` выбирает все тексты, индекс `0` — первый специальный токен `[CLS]`,
последняя ось `:` — все его 768 признаков. После слоёв self-attention этот токен
содержит контекстную информацию о последовательности и даёт фиксированный по длине
вектор для LR. Это соглашение из задания; у базового DistilBERT нет отдельного
обученного sentence-pooling слоя, и `[CLS]` не гарантирует оптимальный вектор текста.

## Воспроизводимость

Запуск: `venv/bin/python lab.py`. Таблица сохранена в `results/metrics.csv`,
метрики с параметрами разбиения и базовым уровнем — в `results/metrics.json`.
Кэш CLS проверяется по SHA-256 текстов с учётом их порядка, имени модели и способу
извлечения. Для принудительного пересчёта используйте `--refresh-features`.

## Документация

- [DistilBERT: входы, attention mask и выходы скрытых слоёв](https://huggingface.co/docs/transformers/model_doc/distilbert).
- [Gensim Downloader: загрузка предобученных векторов](https://radimrehurek.com/gensim/downloader.html).
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    progress(f"Результаты и отчёт сохранены в {output_dir}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16, help="Батч DistilBERT (по умолчанию 16)")
    parser.add_argument("--threads", type=int, default=2, help="Число CPU-потоков PyTorch (по умолчанию 2)")
    parser.add_argument("--device", default="cpu", help="Устройство PyTorch (по умолчанию cpu)")
    parser.add_argument("--refresh-features", action="store_true", help="Пересчитать кэш CLS")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size и --threads должны быть положительными")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA недоступна; используйте --device cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    started = time.perf_counter()
    progress("Задание 1/5. Подготовка данных")
    df = load_data()
    batch_1 = df.iloc[:N_SAMPLES].copy().reset_index(drop=True)
    progress(f"Всего строк: {len(df)}; используем: {len(batch_1)}")
    progress("Распределение классов:\n" + batch_1[1].value_counts().sort_index().to_string())
    train_idx, test_idx = split_indices(batch_1)
    texts = batch_1[0].to_numpy()
    labels = batch_1[1].to_numpy()
    y_train, y_test = labels[train_idx], labels[test_idx]
    records = []

    progress("Задание 2/5. TF-IDF: униграммы и биграммы")
    for name, ngram_range in [
        ("TF-IDF (Unigrams)", (1, 1)),
        ("TF-IDF (N-Grams)", (1, 2)),
    ]:
        vectorizer = TfidfVectorizer(min_df=5, ngram_range=ngram_range)
        train_features = vectorizer.fit_transform(texts[train_idx])
        test_features = vectorizer.transform(texts[test_idx])
        _, record = evaluate_features(name, train_features, test_features, y_train, y_test)
        records.append(record)

    progress("Задание 3/5. Усреднённые векторы GloVe")
    glove_vectors = load_glove()
    glove_features = np.vstack([get_sentence_embedding(text, glove_vectors) for text in texts])
    progress(f"GloVe: матрица признаков {glove_features.shape}")
    _, record = evaluate_features(
        "GloVe (Averaged)", glove_features[train_idx], glove_features[test_idx], y_train, y_test
    )
    records.append(record)
    del glove_vectors, glove_features
    gc.collect()

    progress("Задание 4/5. Контекстные признаки DistilBERT")
    features = extract_bert_features(
        texts, batch_size=args.batch_size, device=args.device,
        refresh=args.refresh_features, threads=args.threads,
    )
    _, record = evaluate_features(
        "DistilBERT (Embeddings + LR)", features[train_idx], features[test_idx], y_train, y_test
    )
    records.append(record)

    progress("Задание 5/5. Сравнение и выводы")
    results = save_results(records, batch_1, train_idx, test_idx)
    progress(results[["method", "train_accuracy", "test_accuracy", "gap"]].to_string(index=False))
    progress(f"Готово. Общее время: {time.perf_counter() - started:.1f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
