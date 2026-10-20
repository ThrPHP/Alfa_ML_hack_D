# Контекст проекта — с чего начать

Папка для того, чтобы новая сессия (или новый человек) за пару минут поняла, где мы находимся.
Подробные технические контракты здесь не дублируются — на них ссылки.

| файл | о чём |
|---|---|
| [case.md](case.md) | суть кейса, ограничения, формула score, инструкция организаторов про ключ и ZIP |
| [team.md](team.md) | кто что делает, репозитории, что кому передать |
| [architecture.md](architecture.md) | карта кода: `ml_core/` и `agent/`, как идёт один прогон |
| [decisions.md](decisions.md) | принятые решения и почему |
| [status.md](status.md) | что сделано, измеренные результаты, известные ограничения |
| [next_steps.md](next_steps.md) | что делать дальше, по порядку |
| [environment.md](environment.md) | облачное окружение: ключ, переменные, установка зависимостей |

Технические контракты:
- [docs/CONTRACTS.md](../docs/CONTRACTS.md) — ml_core (Тима): объекты, операции, гарантии.
- [docs/AGENT_CONTRACT.md](../docs/AGENT_CONTRACT.md) — агент: точка входа `run_research`, Executor, env-переменные.

Исходники кейса: `Base/Альфа Хак_AutoML.pdf`, `Base/baseline.py`, `Base/requirements.txt`, `Base/Datasets/*/task.md`.
