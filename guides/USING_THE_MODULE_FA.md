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

### 3.2 اجرای جداگانهٔ extraction و mapping

هر Capture در این اجرا حداکثر 200 packet دارد. sampler اکنون این packetها را به‌صورت یکنواخت در طول capture انتخاب می‌کند، نه فقط از ابتدای فایل؛ بااین‌حال این اجرا فقط smoke test است و جایگزین اجرای نهایی نیست.

```bash
RUN="artifacts/runs/smoke-all-200"

uv run anomaly dataset extract \
  --output "$RUN" \
  --max-packets 200

# فقط یک‌بار: CSVها را با featureهای ذخیره‌شده مپ و cache کن.
uv run anomaly dataset map "$RUN"
```

هر اجرا باید پوشه‌ی جدید خودش را داشته باشد؛ یک مسیر موجود را برای اجرای تازه بازنویسی نکنید.
`dataset extract` هیچ CSVای را نمی‌خواند. `dataset map` نیز PCAP را باز نمی‌کند؛ فقط
`features/attack/.../records.parquet` و evidence ذخیره‌شده را می‌خواند و خروجی
`labelled/attack/.../records.parquet` را می‌سازد. فرمان `dataset run` همچنان برای
اجرای یک‌خطی extraction+mapping باقی مانده است، اما برای اجرای نهایی همین دو مرحلهٔ جدا توصیه می‌شود.

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

تا زمانی که برای پروتکل موردنظر حداقل یک mapping پذیرفته‌شده **و حداقل یک `attack_records` در `packet_label_coverage` فایل audit** ندارید، Stage 2 را آموزش ندهید. اگر audit فقط `BENIGN` یا `unknown` نشان می‌دهد، PCAP و CSV در بازهٔ حملهٔ قابل مشاهده هم‌پوشانی ندارند؛ آستانه‌ی mapping را برای عبور اجباری کم نکنید.

### نگاشت یک‌بار، استفادهٔ چندباره

اولین `dataset map <dataset-run>` برای هر PCAP/CSV، همهٔ packetهای feature استخراج‌شده را scan می‌کند و
برای هر packet تمام رکوردهای CSV سازگار با endpoint و بازهٔ زمانی را بررسی می‌کند.
نتیجه فقط یک برچسب خلاصه نیست: relation کامل زیر در مسیر سراسری artifact نگه‌داری
می‌شود:

```text
artifacts/mapping_cache/v1/<protocol>/<signature>/
  packet-csv-links.parquet   # packet_uid, packet_index, timestamp, flow_id, csv_row, label, confidence
  flow-evidence.parquet      # evidence و offset در سطح flow
  mapping-cache.json         # snapshot ورودی، policy، acceptance و coverage
```

هر `packet_uid` به‌شکل `<capture-name>:<original-packet-index>` است؛ بنابراین از
خود Parquet مشخص است هر label دقیقاً برای کدام packet PCAP ثبت شده است. در اجرای
بعدی، اگر PCAP، CSV، policy نگاشت و cap تغییر نکرده باشند، اجرای دوبارهٔ `dataset map`
relation را بدون مقایسهٔ مجدد CSV به featureها متصل می‌کند. بعد از یک mapping موفق،
آموزش مدل اصلاً `dataset map` را اجرا نمی‌کند و فقط `labelled/.../records.parquet` را می‌خواند.
ستون‌های `mapping_cache_status` و
`mapping_cache_signature` در `reports/mapping-audit.parquet` نشان می‌دهند cache
`created`، `reused` یا `disabled` بوده است.

افزودن یا تغییر فایل جدید cache همان مورد را خودکار نامعتبر می‌کند. فقط برای ساختن
دوبارهٔ عمدیِ یک ورودی تغییرنکرده از این گزینه استفاده کنید:

```bash
uv run anomaly dataset map artifacts/runs/smoke-all-200 --remap
```

در console و `mapping-progress.json` رویدادهای `mapping_cache_hit`،
`mapping_cache_miss` و `packet_mapping_progress` شمار packetهای بررسی‌شده، packetهای
حمله و batch جاری را نمایش می‌دهند.

## 4. ساخت پروفایل فیچر

بله، از خود داشبورد می‌توان پروفایل جدید ساخت:

1. پروتکل `dns` را از سایدبار انتخاب کنید.
2. به «پروفایل و پیش‌پردازش» بروید.
3. نام ساده، فیچرهای مدل و دلیل انتخاب را وارد کنید.
4. «ذخیره‌ی نسخه‌ی پروفایل» را بزنید.

هر پروفایل immutable و نسخه‌دار است. پروفایل **استخراج PCAP را دوباره اجرا نمی‌کند**؛
یک‌بار extraction کامل، همهٔ featureهای قابل‌استخراج را در Parquet ذخیره می‌کند و profile
در زمان preprocessing/training فقط تعیین می‌کند کدام‌یک وارد مدل شوند. ترتیب، واژگان
دسته‌ای و فیچرهای انتخاب‌شده همراه مدل در `model-contract.json` ذخیره می‌شوند؛ بنابراین
inference دقیقاً با همان قرارداد آموزش اجرا می‌شود.

معادل CLI برای یک پروفایل کوچک DNS:

```bash
uv run anomaly select create dns-compact \
  --protocols dns \
  --features "packet_length,payload_size,payload_entropy,dns_rcode,dns_qname_entropy"
```

پیش از ساخت profile، کیفیت featureهای همان پروتکل را ببینید و فقط featureهای `model_usable`
را انتخاب کنید:

```bash
uv run anomaly select analyze \
  artifacts/runs/smoke-all-200/features/benign/dns/records.parquet \
  --output artifacts/runs/smoke-all-200/reports/features/benign/dns/profile-quality.parquet
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
3. دکمه‌ی «شروع آموزش Stage 1 و Stage 2 در صورت آمادگی» را بزنید.

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

Stage 1 شامل LSTM Autoencoder و Isolation Forest است و تنها با benign آموزش می‌بیند. دادهٔ نرمالِ هر پروتکل به‌ترتیب زمانی به سه بخش جدا تقسیم می‌شود: **آموزش** برای fit detectorها، **calibration** برای تنظیم threshold، و **evaluation دست‌نخورده** برای گزارش FPR. امتیاز دو detector ابتدا نسبت به threshold مستقل خود نرمال می‌شود؛ سپس حداکثرِ نرمال‌شدهٔ آن‌ها با یک threshold مشترک calibrate می‌شود. اگر CSV/PCAP چند packet را صریحاً `BENIGN` برچسب زده باشد، فقط یک برش زمانی کوچک از آن‌ها به calibration threshold افزوده می‌شود؛ هیچ‌کدام وارد fit مدل نمی‌شوند و بقیه در evaluation باقی می‌مانند. بنابراین تصمیم نهایی یک threshold مشترک با هدف پیش‌فرض FPR برابر `0.5%` دارد و دیگر OR دو هشدار مستقل نیست. رأی مستقل هر detector و ستون‌های `stage1_split` و `stage1_metric_evaluation` برای علت‌یابی و بازتولید دقیق ارزیابی در `stage1/scores.parquet` حفظ می‌شود. گزینهٔ `calibrated_weighted` همچنان برای سیاست محافظه‌کارانه‌تر در config موجود است.

Stage 2 یک Random Forest با `balanced_subsample` است و فقط از packetهای حمله با
`mapping_accepted=true` استفاده می‌کند. به‌طور پیش‌فرض حداقل ۲۰ packet، حداقل دو نوع
حمله و حداقل ۴ نمونه از هر نوع لازم است. در صورت امکان یک یا چند **capture کامل** برای
test نگه داشته می‌شود؛ در دیتاست خیلی کوچک که این کار یکی از کلاس‌ها را از train حذف
کند، split stratified در سطح packet انجام می‌شود و این fallback در
`stage2/metrics.json` ثبت می‌شود. فایل `stage2/held-out-split.parquet` دقیقاً مشخص
می‌کند هر packet در train بوده یا test.

اگر این شرایط برقرار نباشد، آموزش خطا نمی‌دهد: Stage 1 ذخیره می‌شود، `stage2/status`
برابر `skipped` و علت در `stage2/readiness.json` ثبت می‌شود. dashboard نیز فقط نتایج
واقعی Stage 1 را نمایش می‌دهد و برای Random Forest معیار ساختگی تولید نمی‌کند.

برای محیطی که false positive از recall مهم‌تر است، فقط همین مقدار را در config اختصاصی پایین‌تر ببرید؛ مدل را دوباره آموزش دهید و هیچ‌وقت threshold قرارداد یک مدلِ موجود را دستی تغییر ندهید:

```yaml
stage_one:
  target_false_positive_rate: 0.002
```

هر مدل در `model-contract.json` mode، وزن‌ها و threshold کالیبره‌شدهٔ خودش را ذخیره می‌کند؛ inference همان تصمیم را تکرار می‌کند.

## 6. اجرای کاملِ مقیاس‌پذیر

بعد از سالم‌بودن اجرای سریع، برای ساخت دیتاست نهایی از همهٔ packetها این فرمان را
بزنید:

```bash
RUN="artifacts/runs/final-20260907"
uv run anomaly --config config/production-streaming.yaml dataset extract \
  --output "$RUN" \
  --all-packets
uv run anomaly --config config/production-streaming.yaml dataset map "$RUN"
```

### مسیر نهایی آموزش، به‌ترتیب درست

پس از دو فرمان بالا، **دیگر `dataset run` را برای همین run اجرا نکنید**. استخراج
و نگاشت کامل شده‌اند و تمام آموزش‌ها فقط `features/...` و `labelled/...` همین run را
می‌خوانند. نخست mapping را در dashboard بررسی کنید، یا فایل‌های زیر را برای ممیزی
نگه دارید:

```text
$RUN/reports/mapping-audit.parquet
$RUN/reports/attack-label-distribution.parquet
$RUN/mappings/<protocol>/<capture>.audit.json
```

سپس برای هر پروتکل یک quality report و یک profile مستقل بسازید. مثال زیر همهٔ
پروتکل‌هایی را که واقعاً benign feature دارند پیمایش می‌کند. فهرست `COMMON_FEATURES`
فقط نقطهٔ شروع مشترک است؛ پس از دیدن `profile-quality.parquet`، featureهای `constant`،
`not_observed` یا `not_implemented` را از profile نهایی خود حذف کنید.

```bash
RUN="artifacts/runs/final-20260907"
COMMON_FEATURES="packet_length,payload_size,payload_entropy,tcp_flag_count,direction,is_request_direction,flow_duration,flow_total_packets,flow_total_bytes,flow_byte_ratio,packet_length_mean,packet_length_std,inter_arrival_time,jitter,packet_rate,hour_sin,hour_cos,weekday_sin,weekday_cos,source_packet_rate,source_destination_count,destination_entropy,burstiness"

for RECORDS in "$RUN"/features/benign/*/records.parquet; do
  [ -f "$RECORDS" ] || continue
  PROTOCOL="$(basename "$(dirname "$RECORDS")")"
  uv run anomaly --config config/production-streaming.yaml select analyze "$RECORDS" \
    --output "$RUN/reports/features/benign/$PROTOCOL/profile-quality.parquet"
  uv run anomaly --config config/production-streaming.yaml select create "$PROTOCOL-core-v1" \
    --protocols "$PROTOCOL" --features "$COMMON_FEATURES" \
    --description "Initial shared CPU-first profile; reviewed against feature quality."
done
```

ساخت profile استخراج PCAP را تکرار نمی‌کند. در preprocessing، featureهای غایب یا
ثابت به‌صورت صریح در `preprocessing.manifest.json` ثبت و حذف می‌شوند؛ profile و
manifest نهایی همراه هر مدل ذخیره می‌شوند. پس از تأیید quality report، برای آموزش
نهایی profile را با همان نام جدید و فقط featureهای `model_usable` دوباره بسازید
(مثلاً `dns-final-v1`) و نام جدید را در فرمان train بدهید.

فرمان زیر برای هر پروتکل یک مدل و یک پوشهٔ artifact جدا می‌سازد. Stage 1 همیشه فقط
با benign آموزش می‌بیند. برای Stage 2، فقط packetهای attack با `mapping_accepted=true`
وارد Random Forest می‌شوند؛ بخشی از آن‌ها برای آزمون نگه داشته می‌شود.

```bash
for RECORDS in "$RUN"/features/benign/*/records.parquet; do
  [ -f "$RECORDS" ] || continue
  PROTOCOL="$(basename "$(dirname "$RECORDS")")"
  uv run anomaly --config config/production-streaming.yaml two-stage train "$RUN" \
    --protocol "$PROTOCOL" --profile "$PROTOCOL-core-v1"
done
```

اگر هر نوع حمله در چند capture باشد، یک یا چند capture کامل برای test کنار گذاشته
می‌شود. اگر این کار پوشش یکی از کلاس‌ها را از train حذف کند، fallback طبقه‌بندی‌شده
در سطح packet به‌کار می‌رود و دلیل آن ثبت می‌شود. مسیرهای مرجع ارزیابی هر مدل:

```text
$RUN/models/<protocol>-<profile>/stage1/metrics.json
$RUN/models/<protocol>-<profile>/stage1/scores.parquet
$RUN/models/<protocol>-<profile>/stage2/readiness.json
$RUN/models/<protocol>-<profile>/stage2/metrics.json
$RUN/models/<protocol>-<profile>/stage2/held-out-split.parquet
$RUN/models/<protocol>-<profile>/pipeline/end-to-end-evaluation.parquet
```

`stage2/held-out-split.parquet` منبع قطعیِ عضویت هر packet در train یا test است.
اگر `stage2/readiness.json` وضعیت آماده‌نبودن را نشان دهد، این شکست آموزش نیست:
مدل به‌شکل Stage 1-only ذخیره می‌شود و prediction ناهنجارِ بدون نوع با
`stage1_anomaly_untyped` مشخص خواهد شد.

`--all-packets` صریحاً extractor جریانی را فعال می‌کند: packetها و فیچرهای همهٔ
captureها در RAM جمع نمی‌شوند؛ هر 50,000 رکورد یک row group فشردهٔ Parquet نوشته
می‌شود و CSV نیز batch-by-batch به همان رکوردهای PCAP نگاشت می‌شود. بنابراین فایل
فیچر کامل جدا برای هر پروتکل/capture ساخته می‌شود، نه یک فایل مشترک. فضای آزاد دیسک
برای artifactها لازم است و زمان اجرا با حجم PCAP متناسب است، اما توقف طولانی بدون
لاگ نباید رخ دهد؛ logهای `STREAM_*` و `DATASET_PROGRESS` مرحله و شمارنده را نشان
می‌دهند.

اگر عمداً فقط یک نمونهٔ بزرگ ولی محدود می‌خواهید، مثلاً سه میلیون packet از هر
capture، از این فرمان استفاده کنید؛ این حالت نیز جریانی و دو-pass است و raw packetها
را در حافظه نگه نمی‌دارد:

```bash
RUN="artifacts/runs/study-3m"
uv run anomaly --config config/production-streaming.yaml dataset extract \
  --output "$RUN" \
  --max-packets 3000000
uv run anomaly --config config/production-streaming.yaml dataset map "$RUN"
```

مدل تمام چندمیلیون رکورد را یکجا در RAM نمی‌خواند. `models.max_source_rows` در
`config/production-streaming.yaml` یک نمونهٔ یکنواختِ سراسری 250,000 رکوردی برای
fit هر پروتکل تعیین می‌کند؛ خودِ corpus کامل روی دیسک باقی می‌ماند. برای افزایش
آموزش، ابتدا 250,000 را به 500,000 بالا ببرید و RAM/زمان را از dashboard یا log
بررسی کنید؛ مقدار `null` برای دیتاست حجیم مسیر امنی نیست. سپس mapping و تحلیل
فیچرها را در dashboard بررسی کنید، پروفایل نهایی را انتخاب کنید و مدل را آموزش دهید:

```bash
uv run anomaly two-stage train \
  artifacts/runs/final-20260907 \
  --protocol dns \
  --profile dns-compact
```

برای اجرای کامل هر چهار پروتکل با profile پیش‌فرض، پس از ساخته‌شدن run این حلقه را اجرا کنید:

```bash
for protocol in dns http modbus s7comm; do
  uv run anomaly --config config/production-streaming.yaml two-stage train \
    artifacts/runs/final-20260907 --protocol "$protocol"
done
```

اگر پروفایل اختصاصی دارید، `--profile <profile-name>` را به همان خط اضافه کنید. داده و مدل هر پروتکل جدا می‌مانند و هیچ فایل feature مشترکی ساخته نمی‌شود.

## 7. آزمون end-to-end پایپ‌لاین

برای ارزیابی مسیر Stage 1 به Stage 2 روی یک جدول فیچر واقعی:

```bash
uv run anomaly two-stage score \
  artifacts/runs/final-20260907/features/attack/dns/browserhijacking-dns/records.parquet \
  --contract artifacts/runs/final-20260907/models/dns-dns-compact/model-contract.json \
  --output artifacts/runs/final-20260907/predictions/dns-browser-hijacking.parquet
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
    stage1/learning-curve.parquet
    stage2/readiness.json
    stage2/metrics.json
    stage2/held-out-split.parquet
    stage2/confusion-matrix.parquet
    stage2/feature-importance.parquet
    stage2/probabilities.parquet
    stage2/permutation-importance.parquet
    stage2/n-estimators-sweep.parquet
    stage2/learning-curve.parquet
    stage2/cross-validation.parquet
    stage2/min-samples-leaf-sweep.parquet
    pipeline/end-to-end-predictions.parquet
    pipeline/end-to-end-evaluation.parquet
    pipeline/end-to-end-probabilities.parquet
    pipeline/ablation-comparison.parquet
    pipeline/inference-latency.parquet
    pipeline/threshold-sensitivity.parquet
    pipeline/seed-stability.parquet
    model-contract.json
    onnx/
```

`catalog/artifacts.parquet` فهرست اصلی همه‌ی جدول‌ها، نمودارهای پشت dashboard و مسیرهای نسخه‌دار است. dashboard چیزی را فقط در حافظه نگه نمی‌دارد؛ همه‌ی تحلیل‌ها به فایل‌های Parquet/JSON قابل استفاده در API، notebook یا سرویس خارجی متکی‌اند.

در `labelled/.../records.parquet` علاوه بر فیچرهای PCAP، ستون‌های `packet_uid` و
`packet_index` برای ارجاع به packet اصلی و ستون‌های `csv_row`، `label_source_file`،
`match_status` و `match_confidence` برای ممیزی نگاشت وجود دارد. relation کم‌حجم و
قابل‌استفاده در سیستم خارجی نیز در `artifacts/mapping_cache/.../packet-csv-links.parquet`
قرار دارد؛ مسیر دقیق آن در audit همان run ثبت می‌شود.

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
uv run anomaly --config config/production-streaming.yaml dataset run \
  --output artifacts/runs/full-server \
  --all-packets
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
