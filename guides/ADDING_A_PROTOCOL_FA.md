# راهنمای افزودن یک پروتکل جدید

این راهنما برای افزودن پروتکل بدون تبدیل‌کردن پروژه به مجموعه‌ای از شرط‌های پراکنده است. هدف این است که هر پروتکل یک extractor، catalog، پوشه‌ی دیتاست، profile و مدل مستقل داشته باشد.

## گام 1: تعریف داده‌ی نمونه و پوشه‌ها

برای پروتکل فرضی `opcua` این ساختار را بسازید:

```text
data/raw/
  benign/pcap/opcua/
  attack/pcap/opcua/
  attack/labels/opcua/
```

حداقل یک PCAP نرمال و چند PCAP/CSV حمله‌ی هم‌زمان داشته باشید. نام فایل‌ها بهتر است مشابه باشند، اما اجباری نیست. فایل CSV باید دست‌نخورده باقی بماند؛ تنها برای label mapping استفاده می‌شود.

## گام 2: فعال‌کردن پروتکل در پیکربندی

در override محیط خود اضافه کنید:

```yaml
capture:
  supported_protocols: [ssh, dns, http, modbus, s7comm, opcua]
```

اگر تشخیص بر اساس پورت ممکن است، `SERVICE_PORTS` و منطق `_infer_protocol` در `src/anomdet/features/extractor.py` را گسترش دهید. اگر پروتکل signature یا layer Scapy دارد، همان را بر پورت ترجیح دهید. تشخیص نباید صرفاً از نام فایل انجام شود.

## گام 3: parser و فیچرها

در `src/anomdet/features/extractor.py` یک تابع کوچک مانند `_extract_opcua(raw)` بسازید که فقط مقادیر واقعاً قابل مشاهده در packet را برگرداند. سپس آن را در `_packet_row` برای `protocol == "opcua"` فراخوانی کنید.

قواعد مهم:

- همواره metadata پایه (`timestamp`, `flow_id`, IP/port، `capture`) را حفظ کنید.
- برای فیلد parse‌نشدنی مقدار ساختگی یا صفر ثابت نسازید. اگر موجود نیست `NaN` بگذارید.
- payload کامل یا شناسه‌ی با cardinality بالا را به‌عنوان فیچر بی‌ملاحظه ذخیره نکنید؛ طول، entropy، count، code و نرخ‌های رفتار معمولاً مناسب‌ترند.
- featureهای رفتاری را در `_augment_features` یا تابع اختصاصی پس از خواندن همه‌ی packetها بسازید تا ترتیب زمان و جریان قابل تکرار بماند.

## گام 4: ثبت feature catalog

در `src/anomdet/features/catalog.py` هر فیچر جدید را با نام snake_case، پروتکل `( "opcua", )`، دسته و توضیح ثبت کنید. تنها featureهای catalog قابل انتخاب در profile هستند؛ بنابراین این مرحله عمداً دروازه‌ی کیفیت است.

برای هر فیچر مشخص کنید آیا واقعاً parser آن را می‌سازد. اگر به decrypt/session reassembly نیاز دارد، آن را در `NOT_YET_IMPLEMENTED_FEATURES` ثبت کنید تا گزارش کیفیت آن را «قابل استفاده» جا نزند.

## گام 5: تعیین خانواده‌ی CSV و mapping aliases

ابتدا header چند CSV را بررسی کنید. اگر با خانواده‌ی IT موجود (Flow ID، Src/Dst IP/Port، Timestamp) یا OT (sIPs/rIPs، startDate/endDate) سازگار است، در pipeline domain را در `_domain` در `src/anomdet/orchestration/dataset_pipeline.py` تعیین کنید.

اگر نام ستون‌ها متفاوت است، aliasها را در `mapping.it` یا `mapping.ot` در override اضافه کنید، برای مثال:

```yaml
mapping:
  it:
    source_ip_columns: [Src IP, src_ip, source_address]
    destination_ip_columns: [Dst IP, dst_ip, destination_address]
    timestamp_columns: [Timestamp, timestamp, event_time]
```

در صورت نیاز به خانواده‌ی واقعاً جدید CSV، یک domain جدید در `normalize_label_csv` اضافه کنید، همراه با تست. مقصد این است که خروجی استاندارد همیشه شامل `label_timestamp`, `src_ip`, `dst_ip`, پورت‌ها (در صورت وجود)، `label`, `csv_row`, `source_file` باشد.

## گام 6: اعتبارسنجی نگاشت قبل از آموزش

فقط ابتدا روی یک فایل کوچک اجرا کنید:

```powershell
uv run anomaly map data/raw/attack/pcap/opcua/sample.pcap data/raw/attack/labels/opcua/sample.csv --domain it --max-packets 5000
```

فایل `.summary.json` و `*.audit.json` را بررسی کنید:

- parse coverage زمان و IPها باید منطقی باشد.
- `match_rate` و `mean_match_confidence` باید از آستانه‌های پیکربندی عبور کنند.
- اگر ساعت PCAP و CSV متفاوت است، offset انتخاب‌شده باید با دانسته‌های dataset سازگار باشد.
- اگر نام‌ها نامرتبط‌اند، `data.attack_pair_overrides.opcua` را اضافه کنید؛ این فقط انتخاب کاندید را تعیین می‌کند و کنترل endpoint/time همچنان اجباری است.

تا وقتی نگاشت پذیرفته نشده، خروجی آن برای Stage 2 استفاده نمی‌شود.

## گام 7: profile و مدل مستقل

پس از اجرای کامل dataset pipeline:

```powershell
uv run anomaly dataset run --max-packets 5000
uv run anomaly select create opcua-core --protocols opcua --features "feature_one,feature_two"
uv run anomaly two-stage train artifacts/runs/<run> --protocol opcua --profile opcua-core
```

هر پروتکل فایل‌های خودش را زیر `features/.../opcua` و `labelled/.../opcua` دارد. مدل `opcua-opcua-core` نیز قرارداد مستقل، preprocessor مستقل و ONNX مستقل دارد. این تفکیک مانع از آن می‌شود که featureهای نامرتبط پروتکل‌های دیگر وارد مدل شوند.

## گام 8: تست‌هایی که باید اضافه شوند

حداقل این سه تست را در `tests/` اضافه کنید:

1. یک packet یا fixture PCAP که اثبات کند protocol درست تشخیص داده و featureهای catalog شده مقدار می‌گیرند.
2. CSV با timestamp/endpoint نمونه که نگاشت صحیح و نگاشت خارج از tolerance را پوشش دهد.
3. profile همان پروتکل که feature نامعتبر را رد می‌کند و قراردادی با ترتیب ثابت می‌سازد.

در پایان اجرا کنید:

```powershell
uv run pytest -q
uv run ruff check src tests
```

## چک‌لیست تحویل

- [ ] پوشه‌های benign/attack/labels پروتکل جدید ساخته شده‌اند.
- [ ] نام پروتکل به `supported_protocols` افزوده شده است.
- [ ] detection، parser و catalog با هم اضافه شده‌اند.
- [ ] CSV به schema استاندارد map می‌شود و audit پذیرفته شده است.
- [ ] profile و model-contract اختصاصی ساخته شده‌اند.
- [ ] نمودارهای feature overview، mapping audit، preprocessing comparison و model metrics در run موجودند.
- [ ] تست‌های parser، mapping و contract سبز هستند.

