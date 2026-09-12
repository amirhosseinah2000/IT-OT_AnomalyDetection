# نقشهٔ ساختار پروژه و توضیح کدها

## هدف و مرزهای سامانه

این مخزن یک سامانهٔ قابل‌حمل و CPU-first برای تشخیص ناهنجاری ترافیک شبکه است.
منبع حقیقت آن فایل‌های محلی Parquet، JSON و مدل است؛ داشبورد فقط لایهٔ نمایش و
اجرای اپراتوری است و داده یا نتیجهٔ مستقل جدیدی تولید نمی‌کند.

```text
PCAP/PCAPNG
  -> استخراج featureهای پروتکل، پکت، flow و رفتار
  -> نگاشت ذخیره‌شدهٔ پکت به CSV، در صورت وجود برچسب
  -> پروفایل feature و قرارداد پیش‌پردازشِ immutable
  -> مرحلهٔ ۱: LSTM-AE + Isolation Forest + گیت اختیاری حملهٔ شناخته‌شده
  -> مرحلهٔ ۲: Random Forest برای نوع حمله
  -> مدل native، ONNX، گزارش‌های Parquet و مصرف در CLI/API/داشبورد
```

دو اصل اصلی در تمام کد رعایت شده است:

1. CSV فقط evidence نگاشت و برچسب است؛ ستون‌های CSV بدون کنترل وارد featureهای
   مدل نمی‌شوند.
2. هر پروتکل، capture، پروفایل feature، قرارداد مدل و خروجی run جدا ذخیره
   می‌شود؛ بنابراین هر سیستم دیگری می‌تواند بدون وابستگی به Streamlit از آن
   استفاده کند.

## ریشهٔ مخزن

| مسیر | شامل چیست و چه می‌کند | آیا باید تغییر کند؟ |
|---|---|---|
| `README.md` | معرفی پروژه، دادهٔ ورودی، دستورات اصلی و quick start. | هنگام تغییر workflow کاربر به‌روز شود. |
| `pyproject.toml` | مشخصات package، نسخهٔ Python، dependencyها، entry point دستور `anomaly`، منبع CPU PyTorch و تنظیم lint/test. | فقط برای packaging یا dependency. |
| `uv.lock` | نسخهٔ دقیق dependencyهای حل‌شده توسط `uv`. | دستی ویرایش نشود؛ با `uv lock`/`uv sync` ساخته شود. |
| `requirements.txt` و `requirements-offline.txt` | فهرست dependency برای نصب معمولی یا آفلاین. | هنگام تغییر dependency. |
| `run_full.txt` | نمونهٔ دستور اجرایی کامل. | کد اجرایی نیست؛ راهنمای عملیاتی است. |
| `dataset-progress.json` | وضعیت آخرین اجرای محلی. | خروجی تولیدشده است، نه source code. |
| `wheelhouse/` و archiveهای `.tar.gz` | بسته‌های نصب آفلاین. | فقط برای انتقال/نصب آفلاین نگه‌داری شوند. |
| `.venv/`، `.uv-cache/` و `.download-venv/` | محیط مجازی و cacheهای local. | تولیدشده‌اند؛ نباید commit یا به‌عنوان source تحویل شوند. |
| `.streamlit/` | تنظیمات محلی Streamlit در صورت نیاز. | فقط برای تنظیم runtime داشبورد. |
| `.gitignore` | تعیین می‌کند داده، artifact و cache وارد Git نشوند. | فقط هنگام تغییر سیاست version-control. |

## تنظیمات: `config/`

تمام YAMLها توسط `anomdet.core.config.load_config` خوانده می‌شوند. این فایل‌ها
ریشهٔ داده، protocolها، محدودیت extraction، سیاست mapping، پیش‌پردازش، CPU/RAM،
معماری مدل، threshold و mirror اختیاری ClickHouse را مشخص می‌کنند.

| فایل | کاربرد دقیق |
|---|---|
| `default.yaml` | تنظیم پیش‌فرض قابل‌حمل تمام commandها، مگر این‌که `--config` داده شود. |
| `production-streaming.yaml` | تنظیم اجرای بزرگ/سروری: extraction streaming روی دیسک و sample محدود برای training. |
| `smoke-4protocol-200.yaml` | تنظیم تست سریع چهار پروتکل با تعداد پکت کم؛ فقط صحت مسیر خروجی را بررسی می‌کند، نه دقت نهایی. |
| `README.md` | توضیح بخش‌های تنظیمات و روش ساخت override امن. |

برای سرور یا آزمایش یک‌بار مصرف بهتر است یک YAML override بسازید؛ override به‌صورت
deep merge با `default.yaml` ترکیب می‌شود و تنظیم اصلی خراب نمی‌شود.

## دادهٔ خام: `data/`

چیدمان استاندارد داده protocol-first است:

```text
data/raw/
  benign/pcap/<protocol>/*.pcap|*.pcapng
  attack/pcap/<protocol>/*.pcap|*.pcapng
  attack/labels/<protocol>/*.csv
```

`<protocol>` فعلاً یکی از `ssh`، `dns`، `http`، `modbus` و `s7comm` است.
PCAPهای benign برای یادگیری Stage 1 هستند. PCAPهای attack فقط منبع feature
هستند؛ تمام پکت‌های آن‌ها خودکار حمله فرض نمی‌شوند. CSV جداگانه با پکت‌ها map
می‌شود تا فقط برچسب‌های قابل اعتماد وارد train/evaluation شوند. مسیرهای
`data/interim` و `data/processed` در صورت وجود، workspace تعریف‌شده در config
هستند و جایگزین قرارداد artifactها نیستند.

## خروجی‌های تولیدشده: `artifacts/`

`artifacts/` ریشهٔ قابل‌انتقال خروجی‌ها است. می‌توان آن را به سرور، API،
ClickHouse یا داشبورد داد. این پوشه source code نیست و معمولاً نباید در Git باشد.

### پروفایل‌های feature

```text
artifacts/feature_profiles/<profile-name>-<content-hash>.json
```

هر فایل JSON نام featureهای خام انتخاب‌شده، پروتکل‌های مجاز، توضیح، زمان ساخت و
نسخهٔ content-hash را نگه می‌دارد. هنگام train، همان profile در model contract
ثبت می‌شود؛ هنگام inference لیست ستون دلخواه کاربر پذیرفته نمی‌شود و transformer
فریز‌شده استفاده می‌شود.

### یک dataset run

```text
artifacts/runs/<run-id>/
  dataset-progress.json
  run-summary.json
  mapping-summary.json                 # پس از `dataset map`
  manifests/input-inventory.json
  catalog/artifacts.parquet
  catalog/summary.json
  features/benign/<protocol>/records.parquet
  features/attack/<protocol>/<capture>/records.parquet
  labelled/attack/<protocol>/<capture>/records.parquet
  mappings/<protocol>/<capture>.parquet
  mappings/<protocol>/<capture>.audit.json
  reports/
  models/<model-run>/<protocol>/
```

| مسیر | دقیقاً چه چیزی در آن است؟ |
|---|---|
| `manifests/input-inventory.json` | نتیجهٔ کشف PCAP و CSV قبل از پردازش. اولین فایل برای اطمینان از دیده‌شدن یک capture جدید است. |
| `catalog/artifacts.parquet` | index ماشین‌خوان همهٔ artifactها: مسیر، نوع، پروتکل، split، توضیح و schema. سیستم‌های خارجی بهتر است فایل‌ها را از این catalog پیدا کنند. |
| `features/benign/<protocol>/records.parquet` | featureهای PCAP-derived با یک رکورد برای هر پکت پشتیبانی‌شدهٔ benign. منبع train نرمال Stage 1 است. |
| `features/attack/<protocol>/<capture>/records.parquet` | featureهای استخراج‌شده از PCAP سمت attack، هنوز بدون برچسب. این فایل به‌تنهایی به معنی «همه پکت‌ها حمله‌اند» نیست. |
| `mappings/...parquet` | evidence نگاشت PCAP/CSV: endpoint/time، تعداد کاندید، confidence، سطر CSV انتخابی و status. |
| `mappings/...audit.json` | کیفیت نگاشت همان capture: time offset، match rate/confidence و دلیل پذیرش یا رد mapping. |
| `labelled/attack/.../records.parquet` | featureهای PCAP پس از الصاق label در زمان هر پکت. فقط `mapping_accepted`، `label` و `is_attack` مشخص می‌کنند یک ردیف حق ورود به supervised evaluation دارد یا نه. |
| `reports/` | گزارش کیفیت feature، مقایسهٔ preprocessing، منابع، توزیع label و گزارش‌های جدولی. |
| `models/` | runهای مدل فریز‌شده. artifact دو مدل از دو پوشهٔ مدل نباید با هم مخلوط شود. |

Cache نگاشت در مسیر تعیین‌شده توسط `mapping.cache_directory` قرار می‌گیرد. این
cache رابطهٔ پکت↔CSV را با کلید هویت PCAP/CSV و policy نگاشت نگه می‌دارد. `dataset
map` بعدی از آن استفاده می‌کند، مگر `--remap` یا تغییر evidence. این cache فقط
نگاشت را ذخیره می‌کند و جای دادهٔ train نیست.

### یک run مدل دو مرحله‌ای

```text
models/<model-run>/<protocol>/
  training-progress.json
  training-summary.json
  model-contract.json
  onnx/
  reports/
  stage1/
  stage2/
  pipeline/
```

| فایل/پوشه | استفادهٔ دقیق |
|---|---|
| `training-progress.json` | event log پایدار برای مانیتور CLI/داشبورد: stage، پیام، progress ratio و جزئیات اجرا. |
| `training-summary.json` | خلاصهٔ مدل‌ها، metricها، منابع، زمان، وضعیت ONNX و مسیر contract. |
| `model-contract.json` | مهم‌ترین ورودی deployment: profile، ترتیب featureهای transform‌شده، preprocessing pipeline، مدل/thresholdهای Stage 1 و وضعیت/مدل Stage 2 را فریز می‌کند. |
| `onnx/` | ONNXهای LSTM-AE، Isolation Forest، RF و در صورت امکان binary gate. فایل `onnx-export.json` نتیجهٔ دقیق export را ثبت می‌کند. |
| `reports/resource-usage.parquet` | snapshot منابع قبل و بعد train. |
| `reports/preprocessing-comparison.parquet` | مقایسهٔ سلامت featureها پیش و پس از preprocessing. |
| `stage1/prepared-normal.parquet` | ماتریس normal آماده‌شده؛ metadata از ستون‌های مدل جدا نگه داشته می‌شود. |
| `stage1/prepared-normal.pipeline.joblib` و `.manifest.json` | pipeline فیت‌شدهٔ imputation/encoding/scaling و manifest فریز؛ برای inference قابل‌حمل لازم‌اند. |
| `stage1/lstm-autoencoder.pt` | state و metadata معماری LSTM-AE در PyTorch. |
| `stage1/isolation-forest.joblib` | مدل native Isolation Forest. |
| `stage1/binary-attack-gate.joblib` | گیت سبکِ حمله‌های شناخته‌شده؛ فقط وقتی labelهای معتبر در captureهای کافی وجود داشته باشند ایجاد می‌شود. |
| `stage1/scores.parquet` | تصمیم پکت‌به‌پکت Stage 1، score، threshold، vote، label قابل‌اعتماد و evaluation-role. |
| `stage1/feature-evidence.parquet` | featureهای دارای بیشترین انحراف robust از normal برای هر رکورد؛ دلیل قابل‌نمایش alert. |
| `stage1/validation-summary.json` | تفکیک تست normal-only، تست capture-held-out با label معتبر، metric توسعه‌ای و مشاهدات mixed بدون برچسب. |
| `stage1/unlabelled-mixed-*.parquet` | score پکت‌های بدون label معتبر. این‌ها فقط مشاهدهٔ عملیاتی‌اند و در accuracy وارد نمی‌شوند. |
| `stage2/attack-random-forest.joblib` | RF تشخیص نوع حمله، فقط در صورت گذر از readiness label. |
| `stage2/readiness.json` و `metrics.json` | توضیح train یا skip امن Stage 2، پوشش class، نوع split و metricها. |
| `pipeline/end-to-end-predictions.parquet` | خروجی گیت Stage 1 و type prediction Stage 2 برای هر پکت evaluation. |
| `pipeline/*.parquet` | تشخیص‌های held-out مانند probability، ablation، حساسیت threshold، latency و پایداری seed، در صورت آماده‌بودن Stage 2. |

## کد نصب‌شونده: `src/anomdet/`

`src/anomdet` خود محصول است. `pyproject.toml` تابع `anomdet.cli:app` را به
command `anomaly` متصل می‌کند. فایل‌های `__init__.py` عمدتاً فقط package marker
هستند، مگر در جایی که API کوچکی export کنند.

### entry point و استفادهٔ خارجی

| فایل | داخل کد چه اتفاقی می‌افتد؟ |
|---|---|
| `cli.py` | adapter خط فرمان با Typer. command را parse می‌کند، config را می‌خواند، ماژول domain صحیح را فراخوانی می‌کند، summary Rich می‌سازد و error را به exit code غیرصفر تبدیل می‌کند. extraction یا math مدل در این فایل پیاده‌سازی نشده است. commandهای مهم: `dataset inspect/run/extract/map`، `select`، `prepare`، `train`، `two-stage train/score`، `dashboard`، `resources` و `replay`. گزینهٔ `--stage-one-only` عمداً RF را رد می‌کند. |
| `service.py` | façade پایتونی برای embed کردن ماژول در سیستم دیگر. `AnomalyService` feature استخراج می‌کند، profile می‌سازد، train/score اجرا می‌کند و از طریق `ModelArtifacts`/`TrainResult` مسیر artifactها را برمی‌گرداند. `quick_anomaly_detection` wrapper سریع است. |
| `analytics/reports.py` | DataFrameهای قابل‌استفادهٔ مجدد برای overview feature و مقایسهٔ raw با preprocessed می‌سازد؛ UI logic ندارد. |

### `core/`: ابزارهای مشترک و امن

| فایل | داخل کد چه می‌گذرد؟ |
|---|---|
| `config.py` | `default.yaml` را می‌خواند، override اختیاری را deep-merge می‌کند و artifact root را مشخص می‌کند. |
| `io.py` | Parquet/CSV/JSONL/JSON را با ایجاد parent directory می‌خواند/می‌نویسد و روش ذخیره‌سازی جدول را یکسان می‌کند. |
| `logging.py` | log ساختاری console/file و ثبت exception با context را تنظیم می‌کند. |
| `paths.py` | درخواست یک PCAP یا پوشه را با کنترل مسیر به یک/چند capture معتبر تبدیل می‌کند. |
| `progress.py` | کلاس `DurableProgress` eventهای JSON پایدار می‌نویسد تا داشبورد یا monitor خارجی بتواند extraction/train طولانی را دنبال کند. |
| `resources.py` | snapshot CPU/RAM می‌گیرد و worker count مؤثر را با رزرو پیش‌فرض یک CPU منطقی می‌سازد. |

### `datasets/`: کشف دادهٔ ورودی

| فایل | داخل کد چه می‌گذرد؟ |
|---|---|
| `inventory.py` | ساختار protocol-first benign/attack PCAP و CSV را کشف می‌کند، allowlist/override اختیاری را اعمال می‌کند، شباهت نام فایل را برای candidate pairing می‌سنجد و `CaptureSource`/`AttackPair` و inventory report تولید می‌کند. |

### `features/`: parsing PCAP و ساخت featureهای causal

| فایل | داخل کد چه می‌گذرد؟ |
|---|---|
| `extractor.py` | موتور اصلی PCAP. با Scapy PCAP/PCAPNG را می‌خواند، protocol را تشخیص می‌دهد، فیلدهای پایهٔ پکت را می‌سازد، state محدود flow/host نگه می‌دارد، featureهای timing/behavior اضافه می‌کند، برای capture بزرگ Parquet batchهای ثابت می‌نویسد، در حالت cap sample قطعی می‌گیرد و progress/manifest ثبت می‌کند. feature یک پکت از future packet استفاده نمی‌کند. |
| `catalog.py` | catalogue ثابت featureها: نام، protocolهای مجاز، category، توضیح و هزینهٔ تقریبی CPU؛ در selection/report/dashboard استفاده می‌شود. |
| `protocols/base.py` | interface parser protocol؛ هر parser packet و raw payload می‌گیرد و فقط fieldهای واقعاً مشاهده‌شده را برمی‌گرداند. |
| `protocols/common.py` | decode امن متن payload و entropy مشترک parserها. |
| `protocols/dns.py` | qname، labelهای DNS، qtype، rcode، TTL، answer count و featureهای lexical/behavior قابل مشاهده را استخراج می‌کند. |
| `protocols/http.py` | request/response line، header، method، status، host/path/user-agent و fieldهای content را parse می‌کند؛ context فقط در همان capture/flow/direction propagate می‌شود. |
| `protocols/modbus.py` | header و function code، unit ID، address/quantity/value و category تابع Modbus TCP را می‌خواند. |
| `protocols/s7comm.py` | fieldهای قابل مشاهدهٔ header/function/parameter/data-size در S7comm را استخراج می‌کند. |
| `protocols/ssh.py` | banner/handshake/algorithmهای SSH غیررمزشده را استخراج می‌کند؛ اطلاعات authentication رمزشده را جعل نمی‌کند. |

### `mapping/`: نرمال‌سازی CSV و labelگذاری evidence-based

| فایل | داخل کد چه می‌گذرد؟ |
|---|---|
| `mapper.py` | schemaهای CSV IT/OT را نرمال می‌کند؛ flowهای PCAP را خلاصه می‌کند؛ candidate سازگار با endpoint می‌سازد؛ offset ساعت CSV/PCAP را تخمین می‌زند؛ قوی‌ترین candidate را انتخاب می‌کند؛ confidence/status می‌نویسد؛ label را در زمان هر پکت attach می‌کند؛ Parquet batch labeling و coverage را انجام می‌دهد؛ و packet-link cache را ذخیره/reuse می‌کند. مقدار `unknown` هرگز به label جعلی تبدیل نمی‌شود. |

### `selection/` و `preprocessing/`: قرارداد feature فریز‌شده

| فایل | داخل کد چه می‌گذرد؟ |
|---|---|
| `selection/profiles.py` | profile JSON با content-hash می‌سازد، profile را resolve/load می‌کند و گزارش کیفیت feature شامل coverage، cardinality، variance، usability و cost تولید می‌کند. |
| `preprocessing/pipeline.py` | فقط روی training rows preprocessing را fit می‌کند: اعتبار featureهای انتخاب‌شده، حذف feature constant/غیرقابل‌استفاده، normalizing missing value، محدودکردن categoryهای زیاد، imputation، robust scaling با `SafeRobustScaler`، one-hot encoding، نوشتن prepared matrix/manifest و reuse دقیق همان transform در inference. |

### `modelling/`: train، validation، score و package مدل

| فایل | داخل کد چه می‌گذرد؟ |
|---|---|
| `detectors.py` | پیاده‌سازی سبک PCA/MLP autoencoder برای experimentهای generic. مسیر production دومرحله‌ای به PCA وابسته نیست. |
| `lstm_autoencoder.py` | autoencoder توالی PyTorch را تعریف می‌کند. اندازهٔ معماری را از تعداد featureهای انتخاب‌شده می‌سازد، روی normal window train می‌کند، با `sequence_groups` مرز capture را حفظ می‌کند، MSE را دوباره به score هر پکت تبدیل می‌کند و state dictionary را save/load می‌کند. |
| `training.py` | موتور experiment unsupervised: input را آماده می‌کند، detectorهای انتخابی را fit می‌کند، tableهای score/metric/importance می‌سازد، all-feature و profile انتخابی را مقایسه می‌کند و LSTM sweep محدود اجرا می‌کند. |
| `random_forest.py` | ماژول Stage 2: readiness label را بررسی می‌کند، matrix Stage 2 (اختیاری با score Stage 1) را می‌سازد، در صورت امکان capture held-out split می‌گیرد، RF balanced train می‌کند، diagnosticهای class/probability/importance/sweep می‌نویسد و attack type پیش‌بینی می‌کند. |
| `two_stage.py` | workflow deployable دومرحله‌ای: sourceهای normal و attack-map‌شده را جدا لود می‌کند؛ roleهای capture-aware برای normal/attack می‌سازد؛ LSTM-AE و Isolation Forest normal-only را fit می‌کند؛ threshold پویا را calibrate می‌کند؛ در صورت کافی‌بودن label گیت حملهٔ شناخته‌شده را با capture جدا train می‌کند؛ evidence پکت‌به‌پکت می‌نویسد؛ Stage 2 را فقط در صورت کافی‌بودن label صدا می‌زند؛ model contract/native/ONNX را ذخیره می‌کند؛ و inference قابل‌حمل `score_two_stage` را پیاده می‌کند. |

### `orchestration/`: اتصال مرحله‌های مستقل

| فایل | داخل کد چه می‌گذرد؟ |
|---|---|
| `dataset_pipeline.py` | workflow جدید protocol-first. `run_dataset_extraction` ابتدا featureهای PCAP را می‌نویسد؛ `run_dataset_mapping` بعداً فقط همان records ذخیره‌شده را map می‌کند؛ `run_dataset_pipeline` convenience composition است. catalog، mapping cache، audit، label distribution و progress در همین فایل به‌روز می‌شوند. |
| `batch.py` | workflow قدیمی/عمومی inventory؛ discovery، extraction، quality analysis، mapping candidate اختیاری و feature evaluation generic را compose می‌کند. |

### `storage/`: ذخیرهٔ محلی و mirror اختیاری ClickHouse

| فایل | داخل کد چه می‌گذرد؟ |
|---|---|
| `artifacts.py` | `ArtifactStore` artifact را در catalog ثبت و Parquet/JSON محلی را به‌عنوان خروجی authoritative می‌نویسد. `NullMirror` حالت پیش‌فرض database-free است؛ `ClickHouseMirror` فقط در صورت config، جدول سازگار می‌سازد و frameها را mirror می‌کند. |

### `dashboard/`: لایهٔ نمایش Streamlit

| فایل | داخل کد چه می‌گذرد؟ |
|---|---|
| `app.py` | entry point Streamlit و explorer قدیمی/اصلی. artifactهای local را با cache-aware reader لود می‌کند، run/protocol را انتخاب می‌کند، EDA/mapping/model/resource نشان می‌دهد، profile می‌سازد و job محدود اجرا می‌کند. |
| `workbench.py` | workbench اصلی فارسی و RTL. تم، dashboard کلی و per-protocol، دیاگرام pipeline مدل، تحلیل Stage 1/2/end-to-end، evidence feature، tableهای mapping/EDA، browser/download فایل، monitor منابع و monitor job زنده را پیاده می‌کند. این فایل artifactها را می‌خواند؛ نتیجهٔ مدل را تغییر نمی‌دهد. |
| `experiment_worker.py` | subprocess worker داشبورد برای train طولانی. training را اجرا و progress ساختاریافته برای UI می‌نویسد. |

## تست خودکار: `tests/`

تست‌ها از داده‌های کوچک synthetic استفاده می‌کنند و نیازی به آرشیو PCAP production ندارند.

| فایل | چه چیزی را بررسی می‌کند؟ |
|---|---|
| `test_protocol_extractors.py` | fieldهای parser پکت/protocol. |
| `test_mapping.py` و `test_dataset_mapping_cache.py` | ruleهای endpoint/time mapping، label پکت و reuse/mismatch cache. |
| `test_dataset_inventory_allowlist.py` و `test_protocol_folders_and_quality.py` | folder discovery، allowlist، inventory و feature quality. |
| `test_selection_and_preprocessing.py` | profile immutable و قرارداد preprocessing. |
| `test_model_experiments.py` | experiment generic و حفظ مرز capture در LSTM. |
| `test_two_stage.py` | قرارداد train/score دومرحله‌ای و گیت capture-disjoint. |
| `test_random_forest.py` | خروجی/readiness مستقل RF Stage 2. |
| `test_dashboard_protocol_sources.py` و `test_experiment_worker.py` | source selection داشبورد و رفتار worker. |

اجرای کامل کنترل‌ها:

```bash
uv run ruff check .
uv run pytest
```

## مستندات: `guides/` و `docs/`

`guides/` شامل راهنماهای فارسی اپراتوری/توسعه است: استفاده از ماژول، افزودن
protocol، انتخاب پویای feature، پردازش feature، داشبورد، metric، تحلیل خروجی و
استفاده در سیستم خارجی. این فایل هم نقشهٔ ساختار فارسی است. `docs/` شامل
قرارداد integration و راهنمای کامل dashboard است. گزارش Word تحویلی در
`guides/technical-system-report-fa.docx` قرار دارد و توسط
`guides/build_technical_report.py` ساخته می‌شود.

## نقشهٔ تغییر امن

- افزودن protocol: parser در `features/protocols/`، ورودی catalogue در
  `features/catalog.py`، supported-protocol config، تست و راهنمای protocol.
- افزودن/تغییر feature خام: extractor/parser + catalogue + quality test؛ label
  و metadata هرگز model column نشوند.
- تغییر مدل: فقط در `modelling/`؛ سازگاری `model-contract.json` حفظ و test متمرکز
  اضافه شود.
- تغییر schema یا مسیر خروجی: `core/io.py`، `storage/artifacts.py`، writerهای
  orchestration/model، readerهای dashboard و مستند integration با هم بررسی شوند.
- تغییر ظاهر: `dashboard/workbench.py` و در صورت نیاز `dashboard/app.py`؛ مقدار
  artifact و محاسبات مدل نباید تغییر کند.
