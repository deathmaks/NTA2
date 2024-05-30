
# Дискретне Логарифмування

Цей проект реалізує два алгоритми для розв'язання задачі дискретного логарифмування: метод повного перебору та алгоритм Сільвера-Поля-Хеллмана.

## Інструкція по запуску

### Встановлення залежностей

Перед запуском програми, встановіть всі необхідні залежності:

```bash
pip install -r requirements.txt
```

### Запуск програми

#### Запуск з вибором алгоритму та введенням параметрів

Для запуску програми з вибором алгоритму та введенням параметрів вручну, виконайте:

```bash
python main.py --algorithm [bruteforce|sph]
```

#### Запуск бенчмарків

Для запуску бенчмарків використовуйте:

```bash
python main.py --benchmark
```

#### Запуск тестування

Для запуску тестування з різною кількістю цифр у параметрі \( p \), використовуйте:

```bash
python main.py --test <number_of_digits_bruteforce> <number_of_digits_sph>
```

### Приклади використання

Для запуску з алгоритмом повного перебору:

```bash
python main.py --algorithm bruteforce
```

Для запуску з алгоритмом Сільвера-Поля-Хеллмана:

```bash
python main.py --algorithm sph
```

Для запуску бенчмарків:

```bash
python main.py --benchmark
```

Для запуску тестування:

```bash
python main.py --test 5 7
```

### Docker

#### Створення Docker образу

Для використання Docker образу, виконайте:

```bash
docker pull anmatos/discrete_log_solver
```

#### Запуск контейнера

Для запуску Docker контейнера з вибором алгоритму, виконайте:

```bash
docker run -it --rm discrete_log_solver --algorithm [bruteforce|sph]
```

Для запуску Docker контейнера з бенчмарком, виконайте:

```bash
docker run -it --rm discrete_log_solver --benchmark
```

Для запуску Docker контейнера з тестуванням, виконайте:

```bash
docker run -it --rm discrete_log_solver --test <number_of_digits_bruteforce> <number_of_digits_sph>
```

## Вимоги до системи

- Python 3.7+
- Docker (для використання контейнера)

## Автор

Анучін Максим ФБ-11
