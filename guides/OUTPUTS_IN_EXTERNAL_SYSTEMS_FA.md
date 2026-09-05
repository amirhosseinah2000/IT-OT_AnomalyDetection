# مستند استفاده از خروجی‌ها و نمودارها در سیستم دیگر

## اصل حمل‌پذیری

هر `dataset run` یک پوشهٔ مستقل است. dashboard فقط خوانندهٔ این فایل‌هاست؛ هیچ تحلیل مهمی فقط در session مرورگر نگه‌داری نمی‌شود. برای انتقال به سرور، API، notebook یا dashboard دیگر، کل پوشهٔ run را منتقل کنید.

```text
<run>/
  catalog/artifacts.parquet        ← فهرست اصلی همهٔ خروجی‌ها
  manifests/                       ← ورودی و provenance
  features/                        ← PCAP-derived feature tables
  mappings/                        ← evidence نگاشت CSV
  labelled/                        ← PCAP features + mapped labels
  reports/                         ← جدول‌های آمادهٔ نمودار
  models/                          ← مدل‌ها، contract، ONNX و نتایج
```

مسیرهای ذخیره‌شده در `catalog/artifacts.parquet` نسبت به root همان run هستند؛ بنابراین با انتقال کل پوشه، مسیرها معتبر می‌مانند.

## نقطهٔ شروع برای سیستم خارجی

ابتدا catalog را بخوانید و بر اساس `kind`، `protocol` و `split` فایل لازم را انتخاب کنید.

```python
from pathlib import Path
import pandas as pd

run = Path("artifacts/runs/full-run")
catalog = pd.read_parquet(run / "catalog/artifacts.parquet")

dns_attack_features = catalog[
    (catalog["kind"] == "labelled_feature_records")
    & (catalog["protocol"] == "dns")
]
for relative_path in dns_attack_features["path"]:
    table = pd.read_parquet(run / relative_path)
    # مصرف در API، notebook یا سیستم ذخیره‌سازی دیگر
```

## نقشهٔ خروجی‌ها

| نیاز سیستم دیگر | فایل یا الگو | محتوای اصلی | نحوهٔ مصرف |
|---|---|---|---|
| یافتن تمام دارایی‌ها | `catalog/artifacts.parquet` | kind، protocol، split، path، rows، schema hash | رجیستری artifact و discovery خودکار |
| بازتولید ورودی | `manifests/input-inventory.json` | فایل‌های واقعی استفاده‌شده و inventory | audit و provenance |
| تحلیل خام نرمال | `features/benign/<protocol>/records.parquet` | packet metadata و فیچرهای PCAP | baseline، EDA، آموزش Stage 1 |
| تحلیل خام حمله | `features/attack/<protocol>/<capture>/records.parquet` | فیچر PCAP بدون اتکا به CSV | تحقیق مستقل و استخراج مجدد |
| اثبات نگاشت | `mappings/<protocol>/<capture>.parquet` | Flow، status، confidence، label source | کنترل کیفیت و review تحلیل‌گر |
| ممیزی نگاشت | `mappings/<protocol>/<capture>.audit.json` | offset زمانی، نرخ تطبیق و کاندیدها | گزارش امنیتی یا نگهداری داده |
| دادهٔ حمله برای مدل | `labelled/attack/<protocol>/<capture>/records.parquet` | فیچر PCAP + label و mapping_accepted | آموزش/ارزیابی Stage 2 |
| تحلیل کیفیت فیچر | `reports/features/**/*-quality.parquet` | coverage، variation، status و reason | انتخاب پویا و dashboard خارجی |
| آمار توزیع فیچر | `reports/features/**/*-overview.parquet` | mean، median، IQR، p01/p99، zero ratio | نمودار و alert آماری |
| وضعیت کل نگاشت | `reports/mapping-audit.parquet` | یک ردیف برای هر Capture | نمودار match rate، confidence و acceptance |
| توازن کلاس | `reports/attack-label-distribution.parquet` | نوع حمله و تعداد رکورد PCAP | sampling و class-balance |
| قرارداد inference | `models/<scope>/model-contract.json` | profile، ترتیب ورودی و دارایی‌های مدل | اجرای امن inference |
| توضیح anomaly | `models/<scope>/stage1/feature-evidence.parquet` | سه فیچر پرانحراف هر رکورد | SOC، triage و explainability |
| خروجی Stage 1 | `models/<scope>/stage1/scores.parquet` | دو score، threshold و decision | alert ناهنجاری و FPR audit |
| خروجی Stage 2 | `models/<scope>/stage2/scores.parquet` | label واقعی، predicted type و correct | ارزیابی تشخیص نوع حمله |
| خروجی یکپارچه | `models/<scope>/pipeline/end-to-end-predictions.parquet` | نتیجهٔ gate و نوع حمله | API پاسخ نهایی یا SIEM |
| مدل‌های قابل‌استقرار | `models/<scope>/onnx/` | LSTM-AE، Isolation Forest و RF | ONNX Runtime یا سرویس جدا |

## استفاده از جدول‌های پشت نمودار dashboard

هر نمودار dashboard را می‌توان از جدول زیر بازسازی کرد:

| بخش dashboard | جدول پایه | نمودارهای اصلی |
|---|---|---|
| نمای کلی | `catalog/artifacts.parquet` + `mapping-audit.parquet` | تعداد خروجی، coverage پروتکل، وضعیت نگاشت |
| داده و نگاشت | `mapping-audit.parquet` + `attack-label-distribution.parquet` + mappingهای Flow | match rate، confidence، offset، balance، matched/unmatched |
| فیچرها | `*-overview.parquet` + `*-quality.parquet` + sample رکوردها | coverage، IQR، zero ratio، correlation، normal-vs-attack |
| پیش‌پردازش | `preprocessing-comparison.parquet` | raw vs prepared برای mean، std و IQR |
| Stage 1 | `scores.parquet` + `feature-evidence.parquet` + history | score scatter، FPR/FNR، loss و شواهد فیچر |
| Stage 2 | `feature-importance.parquet` + `confusion-matrix.parquet` + scores | importance، confusion و accuracy per class |
| پایپ‌لاین | `end-to-end-predictions.parquet` + `resource-usage.parquet` | عبور از گیت، نوع حمله و مصرف RAM |

dashboard در هر بخش امکان دانلود CSV همان جدول تحلیل را نیز می‌دهد. برای سیستم production بهتر است Parquet اصلی را بخوانید تا typeها و metadata حفظ شوند.

## قرارداد API پیشنهادی

برای سرویس آنلاین، response را از `pipeline/end-to-end-predictions.parquet` یا خروجی `anomaly two-stage score` بسازید:

```json
{
  "packet": {
    "timestamp": "2026-01-01T00:00:00Z",
    "protocol": "dns",
    "flow_id": "...",
    "src_ip": "...",
    "dst_ip": "..."
  },
  "stage1": {
    "anomaly": true,
    "normalized_score": 1.42,
    "lstm_score": 34.2,
    "isolation_forest_score": 0.81
  },
  "stage2": {
    "predicted_attack_type": "dns_spoofing"
  },
  "evidence": [
    {"feature": "dns_qname_entropy", "robust_deviation": 4.8},
    {"feature": "payload_entropy", "robust_deviation": 3.1}
  ],
  "model_contract_version": "..."
}
```

برای enrichment پاسخ، `record_id` خروجی Stage 1 را با `feature-evidence.parquet` join کنید. برای نگهداری context شبکه، `flow_id` را با فیچرهای PCAP join کنید.

## اجرای inference بیرون از dashboard

1. PCAP جدید را با extractor همان پروتکل استخراج کنید.
2. contract مدل سازگار را انتخاب کنید.
3. خروجی را در Parquet دریافت یا مستقیماً در سرویس خود بخوانید.

```bash
uv run anomaly protocol extract dns incoming.pcap \
  --output incoming-dns-features.parquet

uv run anomaly two-stage score incoming-dns-features.parquet \
  --contract artifacts/runs/full-run/models/dns-dns-compact/model-contract.json \
  --output incoming-dns-predictions.parquet
```

مدل با contract خودش ترتیب فیچر، preprocessing و شکل ورودی را کنترل می‌کند. هرگز Parquet یک پروتکل را با contract پروتکل دیگر score نکنید.

## ClickHouse و جایگزینی storage

Parquet/JSON همیشه source of truth هستند. اگر ClickHouse فعال باشد، همان جدول‌ها mirror می‌شوند؛ مصرف‌کننده نباید به dashboard وابسته باشد. برای جایگزین‌کردن ClickHouse، یک پیاده‌سازی جدید از `TableMirror` بسازید و مسیرهای artifact محلی را ثابت نگه دارید. به این ترتیب extractor، مدل، API و dashboard تغییر اساسی نمی‌خواهند.
