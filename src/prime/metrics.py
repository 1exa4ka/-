"""Метрики подписки ПРАЙМ.

Пока всё на чистом Python: pandas начинается с темы 03. Функции здесь
принимают список словарей — ровно то, что отдаёт база через fetchall()
и что приходит из внешнего API.
"""

import pandas as pd


def revenue(payments: list[dict]) -> float:
    """Выручка: сумма списаний."""
    return sum(p["amount"] for p in payments)


def churn_rate(subs: pd.DataFrame, start: str, end: str) -> float:
    """Отток за период: доля подписок, активных на start, которые закончились до end.

    Знаменатель — подписки, живые на НАЧАЛО периода: оформленные внутри периода
    не входят, иначе отток зависел бы от продаж этого же месяца (семинар 5).
    """
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    active = subs[(subs["started_at"] < start)
                  & (subs["ended_at"].isna() | (subs["ended_at"] >= start))]
    return (active["ended_at"] < end).sum() / len(active)
