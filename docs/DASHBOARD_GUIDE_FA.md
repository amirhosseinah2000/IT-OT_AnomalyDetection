# راهنمای فارسی داشبورد تشخیص ناهنجاری شبکه

## ۱. محل درست اجرا و ساختار داده

ریشه‌ی درست پروژه این مسیر است:

```text
D:\anomaly-detection\models\1405-05-27\Models
```

دستورهای داشبورد و pipeline باید از همین مسیر اجرا شوند؛ وجود هم‌زمان `pyproject.toml`، `uv.lock`، `config/`، `data/`، `artifacts/` و `src/` نشانه‌ی درست‌بودن مسیر است.

```powershell
cd D:\anomaly-detection\models\1405-05-27\Models
uv run anomaly dashboard
```

PCAPهای ورودی هر پروتکل در زیرپوشه‌ی خود قرار می‌گیرند:

```text
data/raw/pcap/
  dns/
  http/
  modbus/
  s7comm/
```

برای مثال همه‌ی فایل‌های مستقیم `.pcap` و `.pcapng` در `data/raw/pcap/modbus/` متعلق به یک ورودی Modbus هستند. فایل‌های داخل پوشه‌های تو در تو عمداً خوانده نمی‌شوند تا انتخاب داده شفاف و قابل کنترل بماند.

خروجی هر اجرای pipeline در مسیر زیر نوشته می‌شود:

```text
artifacts/runs/<run-id>/
  features/       جدول ویژگی و manifest هر پروتکل
  mapping/        نتیجه‌ی تطبیق CSV با flowها
  reports/        کیفیت و اعتبار استخراج ویژگی‌ها
  experiments/    مدل‌ها، scoreها و مقایسه‌های دستی
```

## ۲. منطق اصلی داشبورد

داشبورد داده‌ی خام PCAP را مستقیماً parse نمی‌کند. ابتدا باید pipeline یا فرمان `extract` یک جدول ویژگی Parquet و manifest آن را تولید کرده باشد. سپس در ستون کناری به‌ترتیب این سه مورد انتخاب می‌شوند:

| کنترل | کاربرد |
|---|---|
| `Artifact directory` | ریشه‌ی خروجی‌ها؛ در اجرای استاندارد `artifacts` است. |
| `Pipeline run` | یک پوشه‌ی اجرا در `artifacts/runs/`. جدیدترین اجرا معمولاً نخست فهرست است. |
| `Analysis protocol` | پروتکل فعال مانند `modbus`. داشبورد مستقیماً فایل `features/modbus-observation.parquet` همان اجرا را می‌خواند، نه جدول تجمیعی همه پروتکل‌ها. |
| `Refresh artefacts` | cache خواندن جدول‌ها و manifestها را خالی می‌کند و فهرست فایل‌ها را دوباره می‌خواند. پس از کپی خروجی یا اجرای جدید از این دکمه استفاده کنید. |

اگر جدول پروتکل بسیار بزرگ باشد، داشبورد حداکثر ۶۰٬۰۰۰ رکوردِ پخش‌شده در طول فایل را برای نمودارهای تعاملی می‌خواند. این فقط نمونه‌ی نمایشی است؛ فایل Parquet اصلی تغییر نمی‌کند و کارت `Source records` تعداد واقعی رکوردهای منبع را نشان می‌دهد. بنابراین مقدارهای روند، توزیع و همبستگی در رابط کاربری نمونه‌محور هستند؛ تعداد ردیف استخراج‌شده و گزارش‌های ذخیره‌شده مقدار کامل دارند.

## ۳. کنترل صحت خوانده‌شدن PCAPهای جدید Modbus

پس از انتخاب `modbus`، زیر نوار انتخاب‌ها عبارت `Active source` نمایش داده می‌شود. در اجرای فعلی باید به فایل زیر اشاره کند:

```text
artifacts/runs/batch-20260819T062945Z/features/modbus-observation.parquet
```

در همان بخش، بازشونده‌ی **PCAP inclusion audit** وجود دارد. این جدول معتبرترین پاسخ به سؤال «آیا فایل جدید واقعاً خوانده شده است؟» است؛ هر سطر دقیقاً یک PCAP فیزیکی است که extractor خوانده است.

| ستون جدول PCAP inclusion audit | معنی |
|---|---|
| `capture` | نام فایل PCAP/PCAPNG واردشده در استخراج. |
| `packets_read` | تعداد کل frameهایی که از فایل خوانده شده‌اند؛ شامل frameهای غیر Modbus نیز هست. |
| `retained_protocol_rows` | تعداد سطرهایی که بعد از تشخیص و فیلتر پروتکل برای پروتکل انتخاب‌شده نگه داشته شده‌اند. |
| `supported_protocol_counts` | شمارش همه پروتکل‌های پشتیبانی‌شده‌ای که parser در آن فایل دیده است؛ به صورت JSON نمایش داده می‌شود. |
| `filtered_protocol_counts` | تعداد بسته‌های یک پروتکل پشتیبانی‌شده‌ی دیگر که چون پوشه/انتخاب فعلی Modbus بوده‌اند از جدول Modbus کنار گذاشته شده‌اند. |

برای نمونه، manifest اجرای فعلی Modbus ثبت می‌کند که ۱۹ PCAP و ۲۱٬۰۷۵٬۳۸۳ رکورد Modbus خوانده شده است. اگر فایل تازه‌ای در پوشه‌ی ورودی کپی کردید اما نامش در این جدول نبود، باید فقط استخراج را دوباره اجرا کنید؛ داشبورد به‌تنهایی نباید PCAP را parse کند.

## ۴. سیاست مدل‌ها

مدل `pca_autoencoder` از همه انتخاب‌های آموزش جدید حذف شده است، زیرا بازسازی خطی PCA برای دنباله‌ی زمانی و رفتار تراکنشی Modbus مناسب نیست. خروجی‌های PCA قدیمی فقط برای مشاهده و مقایسه‌ی تاریخی قابل خواندن‌اند.

مدل اصلی بازسازی، `lstm_autoencoder` است. اصولی که از ریپوی مرجع Modbus در طراحی فعلی استفاده شده‌اند عبارت‌اند از:

- ترتیب زمانی داده در split آموزش/اعتبارسنجی حفظ می‌شود؛ داده‌ی آینده وارد fit نمی‌شود.
- score هر رکورد، **MSE بازسازی** است؛ هرچه بزرگ‌تر باشد رفتار از الگوی آموزش دورتر است.
- آستانه از صدک scoreهای train ساخته می‌شود؛ با `contamination: 0.05` برابر صدک ۹۵ است.
- LSTM دارای validation، early stopping، ثبت history هر epoch و artifact قابل استفاده‌ی بعدی است.
- schema و پیش‌پردازش فقط روی train fit می‌شوند و برای inference همان schema قفل‌شده استفاده می‌شود.

پیش‌فرض‌های پیشنهادی برای شروع آموزش Modbus عبارت‌اند از: `sequence_length=10`، `hidden_size=128`، `latent_size=64`، دو لایه، `dropout=0.2`، حداکثر ۱۰۰ epoch و `patience=15`. این‌ها نقطه‌ی شروع‌اند، نه ادعای قطعی برای همه‌ی دیتاست‌ها؛ sweep باید فقط بعد از بررسی کیفیت ویژگی و یک baseline انجام شود.

pipeline به‌صورت پیش‌فرض دیگر مدل آموزش نمی‌دهد (`feature_evaluation.enabled: false`). پس استخراج، mapping و گزارش کیفیت سبک‌تر تمام می‌شوند و آموزش از بخش **Train and compare** با تصمیم خود اپراتور آغاز می‌شود.

## ۵. بخش Experiment studio

### ۵.۱. Configured pipeline

این فرم اجرای configمحور ingestion را شروع می‌کند.

| کنترل | معنی |
|---|---|
| `Configuration override` | فایل YAML تنظیمات؛ برای داده‌های فعلی `config/test-datasets.yaml`. |
| `New run name` | نام پوشه‌ی خروجی زیر `artifacts/runs/`. باید یکتا باشد. |
| `Optional packet cap per capture` | سقف بسته برای **هر فایل** PCAP. مقدار صفر یعنی تمام بسته‌ها؛ برای آزمون ساختاری کوچک از مقدار مثبت استفاده کنید. |
| `Run configured pipeline` | فقط بعد از submit عملیات extract، mapping و quality report را اجرا می‌کند. آموزش مدل خودکار نیست. |

### ۵.۲. Feature profiles

در این بخش کارفرما می‌تواند هر تعداد مجموعه‌ویژگی تغییرناپذیر بسازد. baseline همه ویژگی‌ها خودکار در هر مقایسه اضافه می‌شود.

| کنترل | معنی |
|---|---|
| `Profile protocol scope` | پروتکلی که profile برای آن معتبر است. هنگام بررسی Modbus فقط `modbus` را نگه دارید. |
| `Features in this profile` | نام ویژگی‌هایی که کارفرما برای آزمایش انتخاب می‌کند. |
| `Profile name` | نام یکتای profile. |
| `Selection rationale` | دلیل انتخاب؛ برای audit و تحویل پروژه ذخیره می‌شود. |
| `Create immutable feature profile` | JSON profile را می‌سازد. برای تغییر، profile تازه بسازید؛ فایل قبلی تغییر نمی‌کند. |

جدول **Available client-selectable features** این ستون‌ها را دارد:

| ستون | معنی |
|---|---|
| `name` | نام فنی و پایدار ویژگی؛ همین مقدار در API و profile ذخیره می‌شود. |
| `display_name` | نام خواناتر برای نمایش. |
| `protocols` | فهرست پروتکل‌هایی که این ویژگی در catalog برای آن‌ها تعریف شده است. |
| `category` | گروه مفهومی مانند `packet`، `flow`، `timing`، `host_behavior` یا `modbus`. |
| `description` | تعریف عملیاتی ویژگی. |
| `cost` | هزینه‌ی تقریبی محاسبه (`low` یا `medium`). |
| `missing_ratio` | نسبت مقدارهای خالی در گزارش کیفیت؛ نزدیک صفر بهتر است. فقط وقتی گزارش quality موجود باشد. |
| `variance` | واریانس مقدارهای مشاهده‌شده؛ صفر معمولاً یعنی ویژگی ثابت است. |
| `estimated_memory_mb` | حافظه‌ی تقریبی موردنیاز آن ستون در جدول استخراج. |

### ۵.۳. Train and compare

این تنها جایی است که آموزش شروع می‌شود و همه ورودی‌ها داخل یک form هستند؛ تغییر sliderها تا فشار دکمه اجرا نمی‌کند.

| کنترل | معنی |
|---|---|
| `Deployment strategy` | `per_protocol` برای مدل کاملاً جدا برای هر پروتکل (انتخاب پیشنهادی برای Modbus)؛ `grouped` برای آموزش مشترک پروتکل‌های IT/OT. |
| `Grouped deployment scope` | فقط در حالت grouped؛ محدوده `all`، `it` یا `ot`. |
| `Detectors` | مدل‌های قابل آموزش جدید. `lstm_autoencoder` مدل بازسازی زمانی است؛ PCA در این فهرست نیست. |
| `Selected feature profiles` | profileهای انتخابی کارفرما؛ baseline همه ویژگی‌ها همیشه مقایسه می‌شود. |
| `Evaluation labels (optional)` | یک mapping بررسی‌شده برای محاسبه ROC-AUC و AP؛ خالی‌گذاشتن آن کاملاً unsupervised است. |
| `Hidden size` | ابعاد حالت پنهان LSTM. بزرگ‌تر معمولاً ظرفیت و هزینه را بالا می‌برد. |
| `Latent size` | اندازه‌ی نمایش فشرده‌ی دنباله. |
| `Sequence length` | تعداد رکوردهای پی‌درپی در هر window. |
| `Epochs` | سقف epoch؛ early stopping ممکن است زودتر متوقف کند. |
| `Batch size` | تعداد window در هر گام optimizer. |
| `Maximum training windows` | سقف windowهای fit برای کنترل زمان و RAM؛ score روی داده‌ی held-out همچنان تولید می‌شود. |
| `Also run an LSTM parameter sweep` | اجرای sweep محدود. |
| `Sweep hidden sizes` | فهرست جداشده با ویرگول از hidden sizeها. |
| `Sweep sequence lengths` | فهرست جداشده با ویرگول از طول دنباله‌ها. |
| `Run experiment` | اجرای مقایسه و نوشتن artifactها در `experiments/`. |

آموزش از جدول همان پروتکل فعال در sidebar شروع می‌شود. برای آموزش Modbus، قبل از بازکردن Experiment studio، `Analysis protocol = modbus` را انتخاب کنید.

### ۵.۴. Execution history

این جدول manifestهای pipeline و experiment را نشان می‌دهد.

| ستون | معنی |
|---|---|
| `manifest` | مسیر نسبی manifest نسبت به run. |
| `created_at` | زمان ساخت artifact به UTC. |
| `detectors` | مدل‌های درخواست‌شده در آن اجرا؛ در pipeline بدون آموزش ممکن است خالی باشد. |
| `runs` | تعداد experiment run یا تعداد datasetهای pipeline. |
| `comparison` | مسیر فایل `comparison.parquet` در صورت وجود. |

## ۶. بخش Results explorer

همه‌ی زیر‌بخش‌های این workspace روی پروتکل فعال کار می‌کنند. فیلتر کردن بعد از خواندن جدول تجمیعی انجام نمی‌شود.

### ۶.۱. Run overview

کارت‌ها:

| کارت | معنی |
|---|---|
| `Source records` | تعداد کامل ردیف‌های جدول Parquet پروتکل، نه فقط نمونه‌ی داشبورد. |
| `Displayed sample` | تعداد رکوردهای نمایش‌داده‌شده؛ برای فایل بزرگ حداکثر ۶۰٬۰۰۰ است. |
| `Flows` | تعداد flowهای یکتای شناسه‌دار در نمونه‌ی نمایشی. |
| `Protocols` | تعداد پروتکل‌های مشاهده‌شده در جدول فعال؛ در حالت جداشده معمولاً ۱ است. |
| `Feature columns` | تعداد ستون‌های عددی قابل تحلیل، بدون شناسه‌ها و portها. |
| `Observed period` | بازه‌ی زمانی نمونه بر اساس timestamp. |

نمودارها: حجم رکورد، timeline دقیقه‌ای، مبدأهای فعال و histogram ویژگی انتخابی. به دلیل نمونه‌برداری، برای شمارش نهایی به manifest و report مراجعه کنید.

### ۶.۲. Protocol explorer

#### Traffic & timing

روند دقیقه‌ای، توزیع فاصله‌ی ورود، jitter، نرخ بسته، حجم بسته، حجم payload، نسبت جریان و شاخص‌های زمانی را فقط برای پروتکل فعال نشان می‌دهد. هدف آن کشف burst، تغییر شیفت زمانی، flow غیرعادی و تغییر ناگهانی load است.

#### Endpoints & flows

نمودارهای مبدأ/مقصد فعال، زوج endpoint، portهای پرتکرار، تعداد بسته و byte هر flow، duration و توزیع‌های flow را نمایش می‌دهد. برای Modbus، endpointهای PLC و HMI و پورت ۵۰۲ باید در این بخش بررسی شوند.

#### Feature health

برای هر ویژگی عددی، coverage، missingness، IQR، variance، نسبت صفر، نرخ رخداد نادر، یکتایی و همبستگی مطلق محاسبه می‌شود. نمودار correlation map فقط ویژگی‌های با signal value بالاتر را نمایش می‌دهد تا نقشه کاربردی بماند.

#### Feature value & guide

`signal_value` یک نمره‌ی عملیاتی ۰ تا ۱۰۰ است، نه اهمیت مدل. این نمره از وجود داده، IQR، کم‌بودن همبستگی زائد و حساسیت به رخداد نادر ساخته می‌شود. اهمیت مدل باید در Model results دیده شود.

جدول **Feature-value evidence table**:

| ستون | معنی |
|---|---|
| `feature` | نام فنی ویژگی. |
| `category` | گروه catalog ویژگی. |
| `signal_value` | امتیاز عملیاتی ۰ تا ۱۰۰ برای اولویت بررسی/انتخاب. |
| `availability` | نسبت ردیف‌های غیرخالی. |
| `iqr` | فاصله چارکی؛ spread مقاوم در برابر outlier. |
| `outlier_rate` | نسبت مشاهده‌های خارج از بازه مقاوم؛ نماینده رخدادهای نادر. |
| `max_abs_correlation` | بیشترین قدرمطلق همبستگی با ویژگی‌های دیگر؛ مقدار بالا نشانه افزونگی است. |
| `cost` | هزینه تقریبی محاسبه. |
| `reason` | توضیح ماشینی اینکه چرا ویژگی ارزشمند، محدود یا افزونه است. |
| `description` | تعریف انسان‌خوان ویژگی از catalog. |

جدول **Protocol feature catalogue**:

| ستون | معنی |
|---|---|
| `name` | نام پایدار ویژگی. |
| `display_name` | عنوان نمایش. |
| `category` | دسته مفهومی. |
| `description` | تعریف ویژگی. |
| `cost` | هزینه محاسبه. |
| `present_in_run` | آیا ستون در schema این run وجود دارد؛ وجود داشتن به معنای متغیر و قابل‌استفاده‌بودن نیست. |

#### Model diagnostics

یک detector artifact ذخیره‌شده را انتخاب می‌کند و نمودار distribution score، ECDF، ترتیب score، rolling score، سهم ویژگی و history LSTM را نشان می‌دهد. برای LSTM، score همان reconstruction MSE است. مدل باید قبلاً با `per_protocol` روی همان پروتکل آموزش داده شده باشد.

جدول **Highest-scoring held-out records**:

| ستون | معنی |
|---|---|
| `timestamp` | زمان رکورد در PCAP، اگر موجود باشد. |
| `anomaly_score` | برای LSTM، MSE بازسازی؛ بزرگ‌تر یعنی ناهنجارتر. |
| `score_percentile` | رتبه صدکی score در مجموعه held-out. |
| `is_anomaly` | آیا score از threshold مدل عبور کرده است. |
| `label` | برچسب تطبیق‌شده‌ی اختیاری؛ `unknown` به‌معنای نبود شواهد کافی است. |
| `flow_id` | شناسه flow دوبخشی و capture-isolated. |
| `row_id` | اندیس پایدار رکورد در خروجی آماده‌شده؛ برای اتصال score به زمینه‌ی ویژگی استفاده می‌شود. |

### ۶.۳. Model results

دو حالت دارد: **Compare models** و **Inspect one detector**. فقط نتیجه‌های ذخیره‌شده خوانده می‌شوند و آموزش دوباره اجرا نمی‌شود.

جدول خلاصه‌ی مقایسه‌ی profileها (All features versus selected profiles) از `comparison.parquet` ساخته می‌شود:

| ستون | معنی |
|---|---|
| `feature_profile` یا `scope` | profile ویژگی یا محدوده‌ی آموزش؛ `all_features` baseline است. |
| `input_features` | تعداد ستون‌های آماده‌شده پس از encoding و حذف ثابت‌ها. |
| `fit_seconds` | زمان fit مدل بر حسب ثانیه. |
| `process_memory_delta_mb` | تغییر RSS پردازه در زمان fit؛ شاخص تقریبی RAM. |
| `predicted_anomaly_rate` | سهم held-out که از threshold بالاتر بوده‌اند. |
| `score_p95` | صدک ۹۵ score. |
| `score_p99` | صدک ۹۹ score؛ ممکن است در جدول کامل باشد. |
| `score_mean` و `score_std` | میانگین و انحراف معیار score. |
| `roc_auc` | فقط با mapping معتبر و دو کلاس؛ کیفیت رتبه‌بندی. |
| `average_precision` | فقط با برچسب معتبر؛ کیفیت تمرکز روی ناهنجاری‌ها. |
| `reconstruction_mse_mean` | میانگین MSE بازسازی LSTM. |
| `reconstruction_rmse` | ریشه MSE بازسازی. |
| `best_training_loss` | کمترین MSE train در epochهای اجراشده. |
| `best_validation_loss` | کمترین MSE validation؛ برای تشخیص overfit مهم است. |
| `feature_reduction_vs_all` | درصد کاهش تعداد ستون آماده‌شده نسبت به baseline همه ویژگی‌ها. |

جدول **Complete comparison table** تمام ستون‌های artifact را بدون خلاصه‌سازی نشان می‌دهد. ستون‌های پایه‌ی آن عبارت‌اند از:

| ستون | معنی |
|---|---|
| `scope` | نام scope مانند `modbus` یا `grouped_ot`. |
| `protocols` | پروتکل‌های واردشده در scope. |
| `model` | نام detector. |
| `training_rows` | تعداد رکوردهای استفاده‌شده برای fit. |
| `test_rows` | تعداد رکوردهای held-out. |
| `input_features` | تعداد ستون آماده‌شده. |
| `fit_seconds` | زمان fit. |
| `process_memory_delta_mb` | تغییر حافظه پردازه. |
| `importance_method` | روش سهم ویژگی؛ برای LSTM `permutation_score_shift`. |
| `threshold` | آستانه تصمیم score. |
| `predicted_anomaly_rate` | نرخ flagشدن held-out. |
| `score_mean`، `score_std`، `score_p50`، `score_p95`، `score_p99` | آمار توزیع score. |
| `roc_auc` و `average_precision` | فقط در صورت اعتبار labelها. |
| `reconstruction_mse_mean` و `reconstruction_rmse` | فقط LSTM. |
| `epochs_completed` | تعداد epochهای واقعاً اجراشده تا early stopping یا سقف. |
| `best_training_loss` و `best_validation_loss` | بهترین lossهای ثبت‌شده‌ی LSTM. |

جدول **Feature contribution** (در artifactهای مدل موجود) این ستون‌ها را دارد:

| ستون | معنی |
|---|---|
| `scope` | scope مدل. |
| `model` | مدل صاحب این اهمیت. |
| `feature` | ویژگی خام catalog. |
| `method` | روش برآورد اهمیت. برای LSTM جابجایی تصادفی یک ستون و سنجش تغییر score است. |
| `raw_importance` | مقدار خام تغییر score؛ فقط داخل همان مدل/داده قابل مقایسه است. |
| `importance` یا `importance_share` | سهم نرمال‌شده بین صفر و یک. |
| `importance_rank` | رتبه نزولی اهمیت. |
| `transformed_columns` | تعداد ستون‌های ایجادشده پس از one-hot که به این ویژگی خام برمی‌گردند. |
| `signal_value` | فقط اگر با گزارش ارزش عملیاتی merge شده باشد؛ اهمیت مدل نیست. |
| `reason` | توضیح evidence عملیاتی، در صورت وجود. |

جدول **Held-out anomaly triage** در artifactهای قدیمی/جمعی می‌تواند این ستون‌ها را داشته باشد:

| ستون | معنی |
|---|---|
| `anomaly_score` | score ناهنجاری. |
| `is_anomaly` | نتیجه threshold. |
| `label` | label اختیاری. |
| `protocol` | پروتکل رکورد. |
| `flow_id` | شناسه flow. |
| `scope` | scope مدل. |
| `model` | detector تولیدکننده score. |

جدول **LSTM parameter sweep** یک ردیف برای هر variant و profile دارد. علاوه بر ستون‌های جدول مقایسه، ستون‌های زیر را اضافه می‌کند:

| ستون | معنی |
|---|---|
| `sweep_variant` | شناسه variant مانند `variant-01`. |
| `hidden_size` | hidden size همان variant. |
| `sequence_length` | طول window همان variant. |
| `sequence_stride` | گام حرکت window. |
| `latent_size` | بعد فضای نهفته، اگر ثبت شده باشد. |
| `epochs`، `batch_size`، `max_train_windows` | پارامترهای هزینه و آموزش variant، اگر ثبت شده باشند. |

### ۶.۴. Mapping audit

mapping برای اعتبارسنجی برچسب است، نه حقیقت خودکار برای آموزش unsupervised. جدول آن:

| ستون | معنی |
|---|---|
| `mapping` | نام فایل summary mapping. |
| `feature_rows` | تعداد سطرهای feature واردشده به تطبیق، اگر summary آن را ثبت کرده باشد. |
| `label_rows` | تعداد سطرهای CSV label، اگر ثبت شده باشد. |
| `matched` | تعداد flow/رکوردی که با شواهد ۵-tuple و زمان تطبیق یافته‌اند. |
| `unmatched` | تعداد موردهای بدون تطبیق قابل اعتماد. |
| `status` | وضعیت اجرای mapping. |

قبل از استفاده از `label` در ROC-AUC یا AP، فایل mapping را باز کنید و نسبت `unmatched` و منطق تطبیق را بررسی کنید.

### ۶.۵. Runtime

کارت‌های `Logical CPUs`، `CPU utilisation`، `Available memory` و `Memory utilisation` snapshot لحظه‌ای ماشین‌اند. اگر comparison موجود باشد، نمودار زمان fit در برابر تغییر RAM را برای هر مدل نشان می‌دهد. این بخش معیار دقت نیست؛ برای ظرفیت‌سنجی اجراست.

## ۷. ترتیب پیشنهادی کار با Modbus

1. از ریشه پروژه dashboard را باز کنید و `Artifact directory = artifacts` بگذارید.
2. جدیدترین `Pipeline run` را انتخاب کنید و سپس `Analysis protocol = modbus`.
3. بازشونده‌ی **PCAP inclusion audit** را باز کنید؛ باید نام همه PCAPهای Modbus جدید را ببینید.
4. در **Protocol explorer → Feature health**، ویژگی‌های `constant`، `not_observed` و coverage پایین را کنار بگذارید.
5. در **Feature value & guide**، `signal_value` را فقط برای ساخت profile اولیه استفاده کنید، نه به‌عنوان importance مدل.
6. در **Feature profiles**، یک یا چند profile Modbus بسازید و دلیل انتخاب را ثبت کنید.
7. در **Train and compare**، `per_protocol` و `lstm_autoencoder` را انتخاب کنید؛ ابتدا یک run با تنظیمات پایه انجام دهید.
8. در **Model results**، loss train/validation، MSE، score distribution، threshold و importance permutation را کنار هم بررسی کنید.
9. فقط اگر baseline پایدار بود، sweep محدود انجام دهید و با profile همه ویژگی‌ها مقایسه کنید.

## ۸. عیب‌یابی سریع

| نشانه | علت محتمل | اقدام |
|---|---|---|
| پروتکل Modbus در sidebar نیست | برای run انتخابی manifest/feature Modbus وجود ندارد. | run درست را انتخاب و `Refresh artefacts` بزنید. |
| فایل Modbus در audit نیست | PCAP بعد از آخرین استخراج اضافه شده یا در زیرپوشه است. | فایل را مستقیم در `data/raw/pcap/modbus/` بگذارید و فقط extraction/pipeline را اجرا کنید. |
| داشبورد کند است | جدول بزرگ است یا tab مدل score بزرگ دارد. | پروتکل درست را انتخاب کنید؛ داشبورد فقط نمونه ۶۰هزارتایی feature را بار می‌کند. |
| LSTM نتیجه ندارد | هنوز experiment جداگانه اجرا نشده یا داده/ویژگی کافی نیست. | per_protocol LSTM را از Train and compare اجرا کنید. |
| ROC-AUC/ AP خالی است | label معتبر با هر دو کلاس وجود ندارد. | mapping audit را بررسی کنید؛ unsupervised score همچنان معتبر عملیاتی است. |
| اهمیت LSTM غیرمنطقی است | feature ثابت، sparse، scale مشکل‌دار یا profile نامناسب است. | Feature health و quality report را اول بررسی و سپس experiment را تکرار کنید. |

