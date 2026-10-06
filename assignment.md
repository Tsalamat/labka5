# Практическое задание: Сравнительный анализ подходов к векторизации текста в задаче бинарной классификации

**Цель работы:** На практике сравнить качество работы классификатора (Логистической регрессии) при использовании различных способов векторизации текста: классических статистических методов (TF-IDF, N-grams), эмбеддингов слов (GloVe) и предобученных языковых моделей (DistilBERT).

---

## Задание 1. Подготовка среды и данных

1. Импортируйте необходимые библиотеки: `numpy`, `pandas`, `torch`, `transformers`, а также модули из `sklearn` (`train_test_split`, `LogisticRegression`, `TfidfVectorizer`, `accuracy_score`).
2. Загрузите датасет SST-2 по ссылке:
`[https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv](https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv)`
3. В целях ускорения вычислений возьмите срез из первых **2000 объектов** (`batch_1`).
4. Исследуйте распределение целевого класса (столбец `1`).

```python
# Заполните пропуски

# 1. Загрузка данных
df = ...

# 2. Выбор первых 2000 строк
batch_1 = ...

# 3. Проверка баланса классов
# Ваш код здесь:

```

---

## Задание 2. Классическая векторизация: TF-IDF и N-граммы

### 2.1 TF-IDF (Unigrams)

1. Разбейте выборку `batch_1` на обучающую (`X_train`, `y_train`) и тестовую (`X_test`, `y_test`) в соотношении 75/25 (`random_state=42`).
2. Обучите `TfidfVectorizer` (с параметрами `min_df=5`, `ngram_range=(1, 1)`) на тренировочной части и примените преобразование к train и test.
3. Обучите `LogisticRegression` и посчитайте `accuracy` на обучающей и тестовой выборках.

```python
# 1. Train/Test split
X_train, X_test, y_train, y_test = ...

# 2. Векторизация TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = ...
# fit и transform:
Xtrain = ...
Xtest = ...

# 3. Обучение классификатора и оценка качества
lr_tfidf = ...
# fit и predict:
...

print("TF-IDF Train Accuracy:", ...)
print("TF-IDF Test Accuracy:", ...)

```

### 2.2 Работа с N-граммами *(Самостоятельно)*

1. Обучите `TfidfVectorizer`, используя биграммы или триграммы (например, `ngram_range=(1, 2)`).
2. Обучите модель Логистической регрессии и сравните метрики с Unigram TF-IDF.

```python
# Ваш код для N-gram TF-IDF здесь:

```

---

## Задание 3. Использование предобученных эмбеддингов слов (GloVe) *(Самостоятельно)*

1. Загрузите предобученные векторы слов `glove-twitter-25` с помощью `gensim.downloader`.
2. Напишите функцию, которая усредняет векторы всех слов в предложении для получения одного вектора текста длины 25.
3. Обучите Логистическую регрессию на полученных векторах и оцените `accuracy`.

```python
import gensim.downloader

# 1. Загрузка векторов
glove_vectors = ...

# 2. Функция для получения вектора предложения (Mean Pooling)
def get_sentence_embedding(sentence):
    # Ваш код: разбиение предложения на слова, получение векторов и усреднение
    pass

# 3. Преобразование текстов, обучение модели и оценка accuracy

```

---

## Задание 4. Извлечение признаков с помощью предобученного DistilBERT

DistilBERT — компактная и быстрая версия BERT от HuggingFace, сохраняющая большую часть его высокой точности. Нам необходимо получить векторы текста длины 768 из последнего скрытого слоя (`[CLS]` токен) и передать их на вход Логистической регрессии.

### 4.1 Загрузка модели и токенизатора

Загрузите модель `DistilBertModel` и токенизатор `DistilBertTokenizer` с весами `'distilbert-base-uncased'`.

```python
# Инициализация модели и токенизатора
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

tokenizer = ...
model = ...

```

### 4.2 Подготовка данных (Токенизация, Паддинг, Маскирование)

1. **Токенизация:** Токенизируйте все предложения в `batch_1[0]` с помощью `tokenizer.encode(..., add_special_tokens=True)`.
2. **Паддинг:** Найдите максимальную длину предложения в батче и дополните все векторные последовательности нулями до этой длины.
3. **Маскирование (Attention Mask):** Создайте бинарную маску (`1` — где есть токен, `0` — где нули паддинга), чтобы BERT игнорировал заполнители.

```python
# 1. Токенизация
tokenized = ...

# 2. Паддинг
max_len = ...
padded = ...

# 3. Маскирование
attention_mask = ...

```

### 4.3 Получение эмбеддингов от DistilBERT

Преобразуйте `padded` и `attention_mask` в PyTorch-тензоры. Пропустите их через модель BERT без вычисления градиентов (`torch.no_grad()`) и извлеките векторы `[CLS]`-токенов (`features`).

```python
input_ids = torch.tensor(...)
attention_mask = torch.tensor(...)

with torch.no_grad():
    last_hidden_states = ...

# Извлечение [CLS] векторов (features)
features = last_hidden_states[0][:, 0, :].numpy()
labels = batch_1[1]

```

### 4.4 Обучение классификатора

Разбейте полученные `features` и `labels` на обучающую и тестовую выборки (`train_test_split`, `random_state=42`) и обучите Логистическую регрессию.

```python
# Train/Test Split
train_features, test_features, train_labels, test_labels = ...

# Обучение и оценка качества
lr_bert = ...
...

print("DistilBERT + LR Train Accuracy:", ...)
print("DistilBERT + LR Test Accuracy:", ...)

```

---

## Задание 5. Итоговый анализ и сравнение результатов

Заполните сводную таблицу результатов и ответьте на контрольные вопросы:

| Подход / Модель | Train Accuracy | Test Accuracy |
| --- | --- | --- |
| **TF-IDF (Unigrams)** |  |  |
| **TF-IDF (N-Grams)** |  |  |
| **GloVe (Averaged)** |  |  |
| **DistilBERT (Embeddings + LR)** |  |  |

### Контрольные вопросы:

1. Какой из методов показал наибольшую точность на тестовой выборке? Почему?
2. В чем главное отличие контекстуализированных эмбеддингов (BERT) от статических (GloVe, Word2Vec)?
3. Какую роль играет `attention_mask` при передаче батча предложений в модель BERT?
4. Почему мы брали именно элемент с индексом `[:, 0, :]` из выходов скрытого слоя BERT для использования в качестве вектора всего текста?