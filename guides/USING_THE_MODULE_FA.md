# راهنمای اجرای سامانه تشخیص ناهنجاری

این پروژه **PCAP-first** است: همه‌ی فیچرها فقط از PCAP/PCAPNG استخراج می‌شوند. CSV هرگز وارد ورودی مدل نمی‌شود؛ نقش آن فقط نگاشت قابل‌ممیزی لیبل حمله و ارزیابی مدل است.

## 1. پیش‌نیاز و قاعده‌ی مهم محیط اجرا

دستورهای زیر را فقط در WSL یا Linux اجرا کنید. محیط مجازی ساخته‌شده در WSL را با PowerShell ویندوز اجرا نکنید؛ هر دو سیستم‌عامل باید محیط مجازی مستقل خود را داشته باشند. برای این پروژه، WSL همان محیط پیشنهادی برای استقرار Linux است.

```bash
cd /mnt/d/anomaly-detection/models/Test/Models

# فقط نصب اول یا پس از تغییر وابستگی‌ها
export UV_LINK_MODE=copy
uv sync --link-mode copy
```

`UV_LINK_MODE=copy` فقط هشدار hardlink روی درایو مشترک Windows/WSL را حذف می‌کند و در منطق داده یا مدل تغییری ایجاد نمی‌کند.

ساختار ورودی باید این باشد:

```text
data/raw/
  benign/pcap/<protocol>/*.pcap|*.pcapng
  attack/pcap/<protocol>/*.pcap|*.pcapng
  attack/labels/<protocol>/*.csv
```

- `benign` فقط داده‌ی یادگیری وضعیت نرمال Stage 1 است.
- در هر پروتکل، PCAPهای حمله و CSVهای همان پروتکل مستقل نگه داشته می‌شوند.
- خروجی فیچرها نیز هرگز در یک فایل مشترک همه‌پروتکلی ادغام نمی‌شود.

## 2. ابتدا موجودی داده را بررسی کنید

```bash
uv run anomaly dataset inspect
```

این فرمان فقط داده را می‌خواند و PCAP/CSVهای کاندید را بر پایه‌ی نام نمایش می‌دهد. این pairing هنوز تأیید نهایی نیست؛ تأیید در dataset run با IP، پورت، زمان، flow و offset زمانی انجام می‌شود.

## 3. اجرای محدود و سریع، به ترتیب

### 3.1 تست extractor مستقل یک پروتکل

ابتدا فقط extractor DNS را روی 200 packet تست کنید:

```bash
uv run anomaly protocol extract dns \
  data/raw/attack/pcap/dns/BrowserHijacking_DNS.pcap \
  --output artifacts/features/smoke/dns/browser-hijacking.parquet \
  --max-packets 200
```

اسکریپت‌های مستقل پروتکل‌ها در این مسیر قرار دارند:

```text
src/anomdet/features/protocols/
  dns.py  http.py  modbus.py  s7comm.py  ssh.py
```

### 3.2 اجرای سریع extraction، mapping و گزارش‌ها

هر Capture در این اجرا حداکثر 200 packet دارد. این اجرای سریع، کل زنجیره را بررسی می‌کند اما جایگزین اجرای نهایی نیست.

```bash
uv run anomaly dataset run \
  --output artifacts/runs/smoke-all-200 \
  --max-packets 200
```

هر اجرا باید پوشه‌ی جدید خودش را داشته باشد؛ یک مسیر موجود را برای اجرای تازه بازنویسی نکنید.

### 3.3 بررسی dashboard قبل از آموزش

```bash
uv run anomaly dashboard
```

در مرورگر ویندوز باز کنید:

```text
http://localhost:8501
```

در سایدبار این موارد را انتخاب کنید:

1. «پوشه‌ی خروجی‌ها»: `artifacts`
2. «اجرای دیتاست»: `runs/smoke-all-200`
3. «پروتکل»: `dns`
4. بخش «داده و نگاشت» را باز کنید و مقدارهای `accepted`، `match_rate`، confidence و offset زمانی را بررسی کنید.
5. بخش «فیچرها» را باز کنید و coverage، variation و نمونه‌ی رکوردهای PCAP را ببینید.

تا زمانی که برای پروتکل موردنظر حداقل یک mapping پذیرفته‌شده ندارید، Stage 2 را آموزش ندهید. آستانه‌ی mapping را برای عبور اجباری کم نکنید؛ CSV، ساعت، endpointها و فایل منتخب را بررسی کنید.

## 4. ساخت پروفایل فیچر

بله، از خود داشبورد می‌توان پروفایل جدید ساخت:

1. پروتکل `dns` را از سایدبار انتخاب کنید.
2. به «پروفایل و پیش‌پردازش» بروید.
3. نام ساده، فیچرهای مدل و دلیل انتخاب را وارد کنید.
4. «ذخیره‌ی نسخه‌ی پروفایل» را بزنید.

هر پروفایل immutable و نسخه‌دار است. ترتیب، واژگان دسته‌ای و فیچرهای انتخاب‌شده همراه مدل در `model-contract.json` ذخیره می‌شوند؛ بنابراین inference دقیقاً با همان قرارداد آموزش اجرا می‌شود.

معادل CLI برای یک پروفایل کوچک DNS:

```bash
uv run anomaly select create dns-compact \
  --protocols dns \
  --features "packet_length,payload_size,payload_entropy,dns_rcode,dns_qname_entropy"
```

پروفایل‌ها در مسیر زیر هستند:

```text
artifacts/feature_profiles/
```

## 5. آموزش سریع مدل دو مرحله‌ای

### از داشبورد

در بخش «مدل‌ها»، پس از انتخاب یک پروتکل مشخص و یک پروفایل سازگار:

1. تعداد mapping پذیرفته‌شده را ببینید.
2. تأیید بررسی mapping و کیفیت فیچرها را تیک بزنید.
3. دکمه‌ی «شروع آموزش LSTM-AE + Isolation Forest + Random Forest» را بزنید.

آموزش در پس‌زمینه اجرا می‌شود، dashboard قابل استفاده می‌ماند و log در پوشه‌ی مدل ذخیره می‌شود. از اجرای تکراری هم‌زمان جلوگیری می‌شود. پس از اتمام، نمودارها و جدول‌ها در همان صفحه ظاهر می‌شوند.

### از CLI

این فرمان دقیقاً همان آموزش dashboard است و برای سرور یا اجرای بدون UI مناسب است:

```bash
uv run anomaly two-stage train \
  artifacts/runs/smoke-all-200 \
  --protocol dns \
  --profile dns-compact
```

`--output` را در اجرای معمول ندهید تا مدل کنار همان dataset run ذخیره شود و dashboard خودکار آن را پیدا کند:

```text
artifacts/runs/smoke-all-200/models/dns-dns-compact/
```

Stage 1 شامل LSTM Autoencoder و Isolation Forest است و تنها با benign آموزش می‌بیند. threshold با validation نرمال و guardrail مبتنی بر MAD برای کنترل FPR کالیبره می‌شود. Stage 2 یک Random Forest با `balanced_subsample` است و فقط از رکوردهای حمله‌ای با mapping پذیرفته‌شده استفاده می‌کند.

## 6. اجرای کامل

بعد از سالم‌بودن اجرای سریع، همان مراحل را بدون محدودیت packet انجام دهید:

```bash
uv run anomaly dataset run \
  --output artifacts/runs/full-20260902
```

این اجرا همه‌ی packetها را می‌خواند و ممکن است با توجه به حجم PCAPها طولانی باشد. سپس mapping و تحلیل فیچرها را در dashboard بررسی کنید، پروفایل نهایی را انتخاب کنید و مدل را آموزش دهید:

```bash
uv run anomaly two-stage train \
  artifacts/runs/full-20260902 \
  --protocol dns \
  --profile dns-compact
```

برای پروتکل‌های دیگر، فقط نام پروتکل و پروفایل سازگار را تغییر دهید؛ داده و مدل هر پروتکل جدا می‌مانند.

## 7. آزمون end-to-end پایپ‌لاین

برای ارزیابی مسیر Stage 1 به Stage 2 روی یک جدول فیچر واقعی:

```bash
uv run anomaly two-stage score \
  artifacts/runs/full-20260902/features/attack/dns/browserhijacking-dns/records.parquet \
  --contract artifacts/runs/full-20260902/models/dns-dns-compact/model-contract.json \
  --output artifacts/runs/full-20260902/predictions/dns-browser-hijacking.parquet
```

برای رکوردی که Stage 1 آن را anomaly تشخیص ندهد، مقدار `predicted_attack_type=normal` مورد انتظار است؛ Stage 2 فقط پس از عبور از gate Stage 1 وارد عمل می‌شود.

## 8. محل خروجی‌ها و استفاده خارج از dashboard

هر dataset run مستقل و قابل‌انتقال است:

```text
<run>/
  manifests/input-inventory.json
  features/benign/<protocol>/records.parquet
  features/attack/<protocol>/<capture>/records.parquet
  mappings/<protocol>/<capture>.parquet
  mappings/<protocol>/<capture>.audit.json
  labelled/attack/<protocol>/<capture>/records.parquet
  reports/mapping-audit.parquet
  reports/attack-label-distribution.parquet
  reports/features/...
  catalog/artifacts.parquet
  models/<protocol>-<profile>/
    stage1/metrics.json
    stage1/scores.parquet
    stage1/feature-evidence.parquet
    stage2/confusion-matrix.parquet
    stage2/feature-importance.parquet
    pipeline/end-to-end-predictions.parquet
    model-contract.json
    onnx/
```

`catalog/artifacts.parquet` فهرست اصلی همه‌ی جدول‌ها، نمودارهای پشت dashboard و مسیرهای نسخه‌دار است. dashboard چیزی را فقط در حافظه نگه نمی‌دارد؛ همه‌ی تحلیل‌ها به فایل‌های Parquet/JSON قابل استفاده در API، notebook یا سرویس خارجی متکی‌اند.

## 9. منابع، سرور و ClickHouse

اجرای پیش‌فرض CPU-first است و dashboard مانیتور زنده‌ی CPU/RAM دارد. برای محدودکردن منابع، یک override مانند `config/server.yaml` بسازید:

```yaml
runtime:
  cpu_workers: 8
  memory_limit_gb: 24
models:
  max_source_rows: 100000
```

سپس گزینه‌ی config را پیش از فرمان اصلی بدهید:

```bash
uv run anomaly --config config/server.yaml dataset run --output artifacts/runs/full-server
```

برای dashboard روی سرور، ترجیحاً آن را فقط روی loopback اجرا کنید و با SSH tunnel باز کنید:

```bash
# روی سرور
uv run anomaly dashboard --host 127.0.0.1 --port 8501

# روی سیستم خودتان
ssh -L 8501:127.0.0.1:8501 user@server
```

فایل‌های محلی Parquet/JSON همیشه source of truth هستند. ClickHouse به‌صورت پیش‌فرض غیرفعال است و فقط mirror اختیاری است؛ نبود ClickHouse نباید اجرای محلی یا قابل‌انتقال سامانه را متوقف کند.

## مستندات تکمیلی

- [انتخاب پویای فیچر](DYNAMIC_FEATURE_SELECTION_FA.md): ساخت، نسخه‌بندی و استفاده از profile در آموزش و inference.
- [مرجع محاسبه و پیش‌پردازش هر ویژگی](FEATURE_PROCESSING_FA.md): برای تک‌تک featureها منبع PCAP، فرمول/شرط استخراج، معنای NaN و تبدیل دقیق آن تا ورودی مدل را توضیح می‌دهد.
- [استفاده از خروجی‌ها در سیستم دیگر](OUTPUTS_IN_EXTERNAL_SYSTEMS_FA.md): catalog، جدول‌های پشت نمودارها، API پیشنهادی و inference بیرون از dashboard.
- [معنی هر نمودار و جدول داشبورد](DASHBOARD_VISUALS_FA.md): منبع، سؤال، روش تفسیر و محدودیت 29 نمودار و 12 جدول تابلوی کل سامانه و صفحات تک‌پروتکل.
- [معیارهای ارزیابی](EVALUATION_METRICS_FA.md): فرمول، محل ذخیره، تفسیر و ترتیب تصمیم برای نگاشت، Stage 1، Stage 2، feature quality و منابع.
