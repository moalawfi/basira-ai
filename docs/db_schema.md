# Database Schema Design (SQLite)

## 1. `results` Table
| Column | Type | Description |
|--- |--- |--- |
| `id` | INTEGER (PK) | Unique identifier |
| `type` | TEXT | Type of analysis performed |
| `input_text` | TEXT | The original text analyzed |
| `output_json` | TEXT | Analytical results in JSON format |
| `timestamp` | DATETIME | Time of creation |

## 2. `logs` Table
| Column | Type | Description |
|--- |--- |--- |
| `id` | INTEGER (PK) | Unique identifier |
| `action` | TEXT | Action performed |
| `status` | TEXT | Success or Failure |
| `error_msg` | TEXT | Error details if failed |
| `timestamp` | DATETIME | Time of logging |

## 3. `analytics_cache` Table
| Column | Type | Description |
|--- |--- |--- |
| `id` | INTEGER (PK) | Unique identifier |
| `metric_name` | TEXT | Name of the metric |
| `metric_value` | REAL | Computed value |
| `computed_at` | DATETIME | Time of computation |