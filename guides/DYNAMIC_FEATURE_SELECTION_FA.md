# مستند سیستم انتخاب پویای ویژگی

## هدف

سیستم ابتدا از PCAP، مجموعهٔ کامل فیچرهای قابل‌استخراج همان پروتکل را می‌سازد. سپس کاربر یک **پروفایل فیچر** ایجاد می‌کند تا مشخص کند کدام زیرمجموعه واقعاً وارد مدل شود. بنابراین استخراج خام با انتخاب ورودی مدل یکی نیست.

```text
PCAP → استخراج فیچرهای پروتکل → Parquet جدا برای هر پروتکل
     → انتخاب پروفایل نسخه‌دار → پیش‌پردازش قراردادمحور
     → Stage 1 و Stage 2
```

## محل تعریف و ذخیره‌سازی

- کاتالوگ فیچرها: `src/anomdet/features/catalog.py`
- پروفایل‌های ایجادشده: `artifacts/feature_profiles/<name>-<version>.json`
- منطق اعتبارسنجی و نسخه‌سازی: `src/anomdet/selection/profiles.py`
- قرارداد نهایی مدل: `<run>/models/<protocol>-<profile>/model-contract.json`

هر پروفایل شامل نام، نسخه، پروتکل‌ها، فهرست مرتب فیچرها، دلیل انتخاب و زمان ایجاد است. `version` از محتوای پروفایل ساخته می‌شود؛ بنابراین تغییر یک فیچر، پروفایل و نسخهٔ جدید ایجاد می‌کند و فایل قبلی تغییر نمی‌کند.

## ساخت پروفایل از dashboard

1. یک dataset run را در سایدبار انتخاب کنید.
2. یک پروتکل مشخص، مانند `dns`، انتخاب کنید.
3. به بخش «پروفایل و پیش‌پردازش» بروید.
4. نام، دلیل انتخاب و فیچرها را وارد کنید.
5. «ذخیرهٔ نسخهٔ پروفایل» را بزنید.
6. نمودار کیفیت فیچرهای انتخاب‌شده را بررسی کنید: coverage، وضعیت extraction و هزینه.
7. در بخش «مدل‌ها» همان پروفایل را برای آموزش انتخاب کنید.

## ساخت پروفایل از CLI

```bash
uv run anomaly select create dns-compact \
  --protocols dns \
  --features "packet_length,payload_size,payload_entropy,dns_rcode,dns_qname_entropy" \
  --description "Balanced DNS profile for CPU-first deployment"

uv run anomaly select list
```

در آموزش، بهتر است نام پروفایل را بدهید:

```bash
uv run anomaly two-stage train artifacts/runs/full-run \
  --protocol dns \
  --profile dns-compact
```

Dashboard برای جلوگیری از انتخاب تصادفی، مسیر نسخهٔ دقیق manifest را به آموزش می‌دهد. در نتیجه مدل دقیقاً به همان snapshot پروفایل وابسته می‌ماند.

## اعتبارسنجی انتخاب

سامانه پیش از ذخیره‌سازی و آموزش این کنترل‌ها را اعمال می‌کند:

1. نام فیچر باید در کاتالوگ معتبر باشد.
2. فیچر باید برای پروتکل انتخاب‌شده قابل‌استفاده باشد.
3. فیچر تکراری حذف می‌شود و ترتیب انتخاب حفظ می‌شود.
4. پروفایل نباید خالی باشد.
5. کیفیت داده بررسی می‌شود: مقدار مشاهده‌شده، صفرهای تقریباً دائمی، تنوع و قابلیت استفادهٔ مدل.
6. هنگام آموزش، ستون‌های گم‌شده با قرارداد preprocessing مدیریت می‌شوند؛ ترتیب ورودی مدل تغییر نمی‌کند.

## ارتباط با دو مرحلهٔ مدل

| مرحله | ورودی | نقش پروفایل |
|---|---|---|
| Stage 1 | فیچرهای PCAP از دادهٔ benign و دادهٔ ارزیابی | پروفایل تعداد و ترتیب ورودی LSTM-AE و Isolation Forest را تعیین می‌کند. |
| Stage 2 | همان فضای transformed + scoreهای Stage 1 | پروفایل پایهٔ فیچرهای Random Forest را تعیین می‌کند؛ دو score مرحلهٔ اول در صورت فعال‌بودن به آن افزوده می‌شوند. |
| Inference | PCAP جدید با همان extractor | `model-contract.json` همان پروفایل و preprocessing آموزش را اعمال می‌کند. |

CSV هرگز جزو profile نیست؛ CSV فقط منبع لیبل پس از نگاشت ممیزی‌شده است.

## توصیه برای انتخاب فیچر

- ابتدا در dashboard، بخش «فیچرها» را ببینید. فیچری که دائماً `zero` است یا coverage بسیار کم دارد، معمولاً انتخاب خوبی نیست مگر دلیل دامنه‌ای داشته باشید.
- برای CPU محدود، ابتدا 5 تا 15 فیچر با coverage و تغییرپذیری مناسب انتخاب کنید.
- برای هر پروتکل پروفایل جدا بسازید؛ `dns-compact` و `modbus-compact` نباید یک نسخه باشند.
- یک پروفایل را پس از ارزیابی مدل تغییر ندهید. پروفایل جدید با نام یا توضیح جدید بسازید تا آزمایش‌ها قابل‌مقایسه بمانند.
- معیار اصلی فقط accuracy نیست: در Stage 1 به FPR، FNR، Recall و نمودار شواهد فیچر نگاه کنید؛ در Stage 2 به weighted F1، confusion matrix و توازن کلاس‌ها نگاه کنید.

## استفاده در سیستم دیگر

سیستم خارجی لازم نیست dashboard را اجرا کند. کافی است profile و model contract را کنار مدل منتقل کند:

```text
artifacts/feature_profiles/<name>-<version>.json
<run>/models/<protocol>-<profile>/model-contract.json
<run>/models/<protocol>-<profile>/stage1/prepared-normal.pipeline.joblib
```

سپس ابتدا PCAP جدید را با extractor همان پروتکل به Parquet تبدیل و فرمان زیر را اجرا کنید:

```bash
uv run anomaly two-stage score new-records.parquet \
  --contract <run>/models/<protocol>-<profile>/model-contract.json \
  --output predictions.parquet
```

جزئیات تمام artifactها و روش استفاده از خروجی‌ها در [OUTPUTS_IN_EXTERNAL_SYSTEMS_FA.md](OUTPUTS_IN_EXTERNAL_SYSTEMS_FA.md) آمده است.
