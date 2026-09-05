# مرجع محاسبه و پیش‌پردازش هر ویژگی

این سند پاسخ دقیق به دو سؤال است: «هر ستون از کدام بخش PCAP ساخته می‌شود؟» و «پس از انتخاب در profile چه تبدیلی روی آن رخ می‌دهد؟». مرجع نهایی همین نسخه از کد در `src/anomdet/features/` و `src/anomdet/preprocessing/pipeline.py` است.

## محدوده و قراردادهای داده

- **منبع feature فقط PCAP/PCAPNG است.** CSV پس از استخراج فقط با زمان، IP، پورت و flow برای افزودن `label` نگاشت می‌شود؛ هیچ ستون CSV به مدل وارد نمی‌شود.
- هر ردیف خروجی یک packet پشتیبانی‌شدهٔ IPv4/IPv6 روی TCP یا UDP است. packetهای غیر IP/TCP/UDP و packetهایی که DNS، HTTP، SSH، Modbus یا S7comm تشخیص داده نشوند، ردیف نمی‌سازند.
- تشخیص به‌ترتیب DNS (لایهٔ Scapy یا پورت 53)، Modbus (502)، S7comm (102 با محتوای مناسب)، SSH (22 یا `SSH-`) و HTTP (پورت‌های 80/8000/8080/8081/8888 یا امضای HTTP) است.
- parserها **TCP reassembly ندارند**؛ بنابراین فقط bytes همان segment دیده می‌شود. متن خام با `latin-1` از حداکثر 2048 byte اول خوانده می‌شود. banner SSH تا 255 و name-listهای SSH تا 1024 نویسه محدودند.
- مقدار parse‌نشده همیشه **NaN/خالی** است، نه صفر. صفر تنها هنگامی نوشته می‌شود که معنی صریح دارد؛ مثلاً payload خالی یا تعداد TCP flag برابر صفر. این تفاوت در quality report و imputation مهم است.

خروجی خام به‌صورت جدا نگه‌داری می‌شود:

```text
<run>/features/benign/<protocol>/records.parquet
<run>/features/attack/<protocol>/<capture>/records.parquet
<run>/labelled/attack/<protocol>/<capture>/records.parquet
```

## نمادها و گروه‌بندی

| نماد | تعریف |
|---|---|
| `p` | packet/ردیف فعلی |
| `B(p)` | `packet_length = len(packet)`؛ طول کامل frame مشاهده‌شده در Scapy |
| `R(p)` | bytes لایهٔ Raw؛ اگر Raw نباشد byte خالی |
| `H(x)` | آنتروپی شانون `-Σ pᵢ log2(pᵢ)` روی فراوانی byteها/نویسه‌ها؛ برای ورودی خالی صفر |
| `f` | جریان دوطرفه `(capture, flow_id)` |
| `s` | گروه مبدأ `(capture, src_ip)` |
| `t(p)` | timestamp UTC packet |

`flow_id` از 5-tuple دوطرفهٔ مرتب‌شدهٔ لغوی ساخته و با نام capture پیشوند می‌شود. پس `direction=1` یعنی endpoint مبدأ packet، اولین endpoint در این ترتیب لغوی است؛ client/server نیست. `is_request_direction=1` وقتی است که مقصد packet پورت سرویس شناخته‌شده باشد.

ستون‌های `capture`، `timestamp`، `transport`، `flow_id`، IPها و پورت‌ها برای ردیابی و نگاشت هستند و مستقیماً انتخاب‌پذیر نیستند. `protocol` در آماده‌سازی خودکار اضافه می‌شود تا دادهٔ چندپروتکلی encode شود، اما در اجرای تک‌پروتکل معمولاً ثابت است و حذف خواهد شد.

## ویژگی‌های مشترک packet، flow و زمان

برای این محاسبات، کل داده ابتدا با `(capture, flow_id, timestamp)` مرتب می‌شود. آمار flow بر کل flow مشاهده‌شده در همان capture هستند، نه فقط اطلاعات گذشتهٔ packet.

| ویژگی | نوع خام | محاسبهٔ دقیق |
|---|---|---|
| `packet_length` | عددی | `B(p)=len(packet)`؛ طول frame کامل مشاهده‌شده. |
| `payload_size` | عددی | `len(R(p))`؛ اگر Raw وجود نداشته باشد 0. |
| `payload_entropy` | عددی | `H(R(p))` روی byteهای payload، گرد شده تا 4 رقم؛ payload خالی = 0. |
| `tcp_flag_count` | عددی | برای TCP تعداد bitهای یک در `TCP.flags`؛ برای UDP دقیقاً 0. |
| `direction` | Boolean/عددی | 1 اگر `(src_ip,src_port)` اولین endpoint tuple مرتب‌شده باشد؛ در غیر این صورت 0. |
| `is_request_direction` | Boolean/عددی | 1 اگر `dst_port` پورت سرویس همان پروتکل باشد (HTTP: یکی از پورت‌های HTTP تعریف‌شده)، وگرنه 0. |
| `flow_duration` | عددی، ثانیه | `max(t(p)-min(t(q), q∈f), 0)`. |
| `flow_total_packets` | عددی | تعداد همهٔ packetهای `f`، یعنی `|f|`؛ برای تمام ردیف‌های flow یکسان است. |
| `flow_total_bytes` | عددی | `Σ B(q)` برای همهٔ `q∈f`؛ برای تمام ردیف‌های flow یکسان است. |
| `flow_byte_ratio` | عددی | مجموع `B` در direction فعلیِ `f` تقسیم بر مجموع `B` جهت مخالف؛ مخرج حداقل 1 است. |
| `packet_length_mean` | عددی | میانگین `B(q)` برای کل `f`. |
| `packet_length_std` | عددی | انحراف معیار نمونه‌ای `B(q)` در کل `f`؛ flow تک‌packet/NaN به 0 تبدیل می‌شود. |
| `inter_arrival_time` | عددی، ثانیه | `t(p)-t(packet قبلی همان f)`؛ اولین packet = 0 و مقدار منفی احتمالی به 0 clip می‌شود. |
| `jitter` | عددی، ثانیه | قدر مطلق تغییر `inter_arrival_time` نسبت به packet قبلی همان `f`؛ اولین مقدار = 0. |
| `packet_rate` | عددی، packet/s | `شمارهٔ packet در f از 1 / max(flow_duration,1)`. |
| `hour_sin` | عددی | `sin(2π × hour_UTC(t(p)) / 24)`. |
| `hour_cos` | عددی | `cos(2π × hour_UTC(t(p)) / 24)`. |
| `weekday_sin` | عددی | `sin(2π × dayofweek_UTC(t(p)) / 7)`؛ Monday=0. |
| `weekday_cos` | عددی | `cos(2π × dayofweek_UTC(t(p)) / 7)`؛ Monday=0. |

## ویژگی‌های رفتار مبدأ

| ویژگی | نوع خام | محاسبهٔ دقیق |
|---|---|---|
| `source_packet_rate` | عددی، packet/s | در `s`: `(شمارهٔ ردیف در گروه + 1) / max(t(p)-min(t(q),q∈s),1)`. شمارهٔ ردیف تابع ترتیب داخلی `(capture,flow_id,timestamp)` است. |
| `source_destination_count` | عددی | تعداد `dst_ip` یکتای کل `s` با `nunique`؛ rolling window ندارد. |
| `destination_entropy` | عددی | IPهای مقصد یکتای `s` به متن تبدیل، لغوی مرتب و با `|` وصل می‌شوند؛ سپس `H(متن حاصل)` محاسبه می‌شود. این آنتروپی فراوانی مقصدها نیست. |
| `burstiness` | عددی | `source_packet_rate / max(capture_packet_rate,1e-6)`؛ `capture_packet_rate = تعداد ردیف capture / max(زمان آخر-اول capture,1)`. |

`behavior_window_seconds` با مقدار پیش‌فرض 60 در raw output ثبت می‌شود، اما در پیاده‌سازی فعلی rolling window اعمال نمی‌کند. featureهای رفتار مبدأ و رفتار پروتکل‌ها روی کل capture/group هستند؛ این ستون فقط provenance است و در Feature Catalog انتخاب‌پذیر نیست. اگر محصول به window واقعی 60 ثانیه نیاز دارد، extractor و نسخهٔ schema باید تغییر کنند.

## DNS

فقط با وجود لایهٔ DNS استخراج می‌شود. qname پس از decode و `rstrip('.')` استفاده می‌شود.

| ویژگی | نوع خام | محاسبهٔ دقیق |
|---|---|---|
| `dns_qname` | متنی | `dns.qd.qname` پس از `safe_text` و حذف نقطهٔ انتهایی. |
| `dns_qname_length` | عددی | `len(dns_qname)`. |
| `dns_qname_entropy` | عددی | `H(dns_qname)`. |
| `dns_label_count` | عددی | `تعداد '.' در qname + 1`. |
| `dns_qtype` | عددی | `int(dns.qd.qtype)` در صورت وجود qname. |
| `dns_digit_ratio` | عددی | `تعداد char.isdigit() / max(len(qname),1)`. |
| `dns_hyphen_ratio` | عددی | `تعداد '-' / max(len(qname),1)`. |
| `dns_ngram_score` | عددی | `تعداد char.isalpha() / max(len(qname),1)`، گرد به 4 رقم. نام ستون تاریخی است؛ n-gram واقعی نیست. |
| `dns_rcode` | عددی | `int(dns.rcode)` از header؛ در لایهٔ DNS بدون مقدار صریح، 0. |
| `dns_ttl` | عددی | `int(dns.an.ttl)` از **اولین** answer در صورت قابل‌خواندن بودن؛ وگرنه NaN. |
| `dns_answer_count` | عددی | `int(dns.ancount)` از header. |
| `dns_response_size` | عددی | `len(R(p))` فقط وقتی `dns.qr=1` باشد؛ query = NaN. |
| `dns_authoritative` | Boolean/عددی | `int(dns.aa)`. |
| `dns_recursion_available` | Boolean/عددی | `int(dns.ra)`. |
| `dns_truncated` | Boolean/عددی | `int(dns.tc)`. |
| `dns_nxdomain_rate` | عددی | در `(capture,src_ip)` و ردیف‌های DNS: میانگین شرط `dns_rcode.fillna(0)==3`. |
| `dns_query_repeat_count` | عددی | تعداد ردیف‌ها در `(capture,src_ip,dns_qname)` با `transform('size')`؛ شمارش packet است، نه transaction reassembled. |

## HTTP

parser متن 2048 byte اول payload را بدون TCP reassembly می‌خواند. headerها تا اولین خط خالی خوانده می‌شوند، کلیدشان lowercase است و header تکراری با آخرین مقدار overwrite می‌شود.

| ویژگی | نوع خام | محاسبهٔ دقیق |
|---|---|---|
| `http_method` | متنی | method خط اول request، فقط یکی از `GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS, CONNECT`. |
| `http_url_length` | عددی | طول token مقصد request بعد از method. |
| `http_url_entropy` | عددی | `H(target)` برای مقصد request. |
| `http_query_parameter_count` | عددی | اگر `?` موجود باشد تعداد `=` در بخش پس از اولین `?`، وگرنه 0؛ parse کامل query نیست. |
| `http_suspicious_token_count` | عددی | تعداد tokenهای متمایز موجود در target lowercase از این موارد: `../`، `%2e%2e`، `<script`، `%3cscript`، `union select`، ` or 1=1`، `;--`. هر token حداکثر یک‌بار شمرده می‌شود. |
| `http_host` | متنی | مقدار header `Host` بعد از trim. |
| `http_user_agent` | متنی | مقدار header `User-Agent` بعد از trim. |
| `http_header_count` | عددی | تعداد headerهای دارای `:` تا خط خالی؛ header تکراری یک‌بار در dict باقی می‌ماند. |
| `http_content_length` | عددی | تبدیل عددی `Content-Length`؛ نامعتبر/نبود header = NaN. |
| `http_cookie_count` | عددی | header `Cookie` با `;` شکسته می‌شود و بخش‌های دارای `=` شمرده می‌شوند. |
| `http_status_code` | عددی | کد سه‌رقمی پاسخ مطابق `HTTP/x.y NNN`. |
| `http_response_size` | عددی | `len(R(p))` فقط برای پاسخ تشخیص‌داده‌شده. |
| `http_error_ratio` | عددی | در `(capture,src_ip)`، نسبت statusهای غیرخالی در `[400,600)` به همهٔ statusهای غیرخالی؛ اگر هیچ statusی نباشد NaN. |
| `http_post_get_ratio` | عددی | در `(capture,src_ip)`، `تعداد POST / max(تعداد GET,1)` بر پایهٔ methodهای parse‌شده. |
| `http_request_repeat_count` | عددی | تعداد ردیف‌ها در `(capture,src_ip,http_url)` با `transform('size')`. |

`http_url=target[:2048]` یک ستون کمکی خام است که برای repeat-count لازم است، اما قابل انتخاب در profile نیست. پس از محاسبهٔ رفتارها، تمام HTTP featureهای جدول بالا در هر `(capture,flow_id,direction)` با `ffill().bfill()` پخش می‌شوند. پس context درخواست فقط به packetهای همان جهت flow منتقل می‌شود، نه جهت پاسخ.

## SSH

فقط نشانه‌های قابل‌مشاهده پیش از رمزنگاری استخراج می‌شود.

| ویژگی | نوع خام | محاسبهٔ دقیق |
|---|---|---|
| `ssh_client_banner` | متنی | اگر payload با `SSH-` شروع و `dst_port=22` باشد: اولین line پس از `safe_text`، حداکثر 255 نویسه. |
| `ssh_server_banner` | متنی | همان banner وقتی مقصد پورت 22 نباشد. |
| `ssh_kex_algorithms` | متنی | اولین name-list KEXINIT. parser اولین `0x14` را یافته، 17 byte جلو می‌رود و length-prefixed listها را می‌خواند. |
| `ssh_cipher_count` | عددی | تعداد مقادیر جداشده با `,` در name-list دوم KEXINIT؛ نبود/خالی = 0. |
| `ssh_mac_count` | عددی | تعداد مقادیر name-list چهارم KEXINIT؛ نبود/خالی = 0. |
| `ssh_compression_count` | عددی | تعداد مقادیر name-list ششم KEXINIT؛ نبود/خالی = 0. |
| `hassh` | متنی | MD5 hex از `';'.join(حداکثر 10 name-list قابل‌خواندن)`. اثرانگشت implementation-specific است، نه ادعای decode کامل SSH. |
| `ssh_auth_method` | — | هنوز پیاده‌سازی نشده؛ NaN schema placeholder است و نباید انتخاب شود. |
| `ssh_failed_auth_count` | — | هنوز پیاده‌سازی نشده؛ همیشه NaN. |
| `ssh_open_channel_count` | — | هنوز پیاده‌سازی نشده؛ همیشه NaN. |
| `ssh_keepalive_interval` | — | هنوز پیاده‌سازی نشده؛ همیشه NaN. |

اگر KEXINIT ناقص/نامعتبر باشد، فقط featureهایی که پیش از آن واقعاً قابل مشاهده بوده‌اند ثبت می‌شوند.

## Modbus/TCP

decoder از byte 0 payload MBAP را فرض می‌کند و حداقل 8 byte می‌خواهد؛ TCP reassembly ندارد. تمام integerهای چندبایتی big-endian هستند.

| ویژگی | نوع خام | محاسبهٔ دقیق |
|---|---|---|
| `modbus_transaction_id` | عددی | bytes `0:2` raw، unsigned big-endian. |
| `modbus_unit_id` | عددی | raw byte `6`. |
| `modbus_function_code` | عددی | `raw[7] & 0x7f`؛ bit exception حذف می‌شود. |
| `modbus_function_category` | متنی | `read` برای `{1,2,3,4,7,11,12,17,20,24,43}`؛ `write` برای `{5,6,15,16,21,22,23}`؛ `diagnostic` برای `{8}`؛ وگرنه `other`. |
| `modbus_protocol_id_valid` | Boolean/عددی | 1 اگر bytes `2:4` برابر 0 باشد، وگرنه 0. |
| `modbus_length_valid` | Boolean/عددی | 1 اگر MBAP length در bytes `4:6` برابر `len(raw)-6` باشد. |
| `modbus_nonstandard_function` | Boolean/عددی | 1 اگر function base خارج از `{1..24,43}` باشد. |
| `modbus_exception_code` | عددی | اگر `raw[7]` bit `0x80` داشته باشد و raw بیش از 8 byte باشد: raw byte `8`. |
| `modbus_starting_address` | عددی | برای functionهای `{1,2,3,4,5,6,15,16,22,23}` و raw حداقل 12 byte: bytes `8:10`. |
| `modbus_quantity` | عددی | در همان شرط: bytes `10:12`. |
| `modbus_byte_count` | عددی | اگر raw بیش از 8 byte باشد: raw byte `8`. توجه: برای exception همان exception code و برای بعضی messageها نخستین byte داده است؛ همیشه byte-count استاندارد نیست. |
| `modbus_value` | عددی | برای functionهای `{3,4,6,16}` و raw حداقل 10 byte: **دو byte آخر raw payload**. این parser اولین register را جداگانه استخراج نمی‌کند. |
| `modbus_exception_rate` | عددی | میانگین feature کمکی `modbus_is_exception` در `(capture,flow_id)`؛ helper برابر `int(raw[7] & 0x80 != 0)` است. |
| `modbus_address_deviation` | عددی | `abs(starting_address - median(starting_address در capture,flow_id))`. |
| `modbus_response_matched` | Boolean/عددی | 1 اگر `(capture,flow_id,transaction_id)` دست‌کم دو `direction` یکتا داشته باشد؛ تطبیق جهت/شناسه است، نه صحت کامل پاسخ. |

## S7comm

parser نخستین marker `0x32` را در payload می‌یابد. اگر marker نباشد یا 10 byte پس از آن موجود نباشد، تنها feature قابل تولید `s7_is_plus` است. TCP reassembly و decode کامل data-item انجام نمی‌شود.

| ویژگی | نوع خام | محاسبهٔ دقیق |
|---|---|---|
| `s7_rosctr` | عددی | raw byte `marker+1`. |
| `s7_pdu_reference` | عددی | raw bytes `marker+2:marker+4`، big-endian. |
| `s7_parameter_length` | عددی | raw bytes `marker+4:marker+6`، big-endian. |
| `s7_data_length` | عددی | raw bytes `marker+6:marker+8`، big-endian. |
| `s7_error_class` | عددی | raw byte `marker+8`. |
| `s7_error_code` | عددی | raw byte `marker+9`. |
| `s7_is_plus` | Boolean/عددی | 1 اگر byte string `S7comm-Plus` در raw باشد، وگرنه 0. |
| `s7_function_code` | عددی | byte اول parameter؛ parameter از `marker+10` با طول اعلام‌شده بریده می‌شود. |
| `s7_item_count` | عددی | byte دوم parameter اگر وجود داشته باشد؛ parameter تک‌byte = 0؛ parameter خالی = NaN. |
| `s7_control_command` | Boolean/عددی | 1 اگر function parameter یکی از `0x28,0x29` باشد؛ وگرنه 0. |
| `s7_block_transfer` | Boolean/عددی | 1 اگر function parameter یکی از `0x1A,0x1B` باشد؛ وگرنه 0. |
| `s7_transfer_size` | عددی | فقط اگر parameter حداقل 10 byte و function `0x04` یا `0x05` باشد: `parameter[4]`. |
| `s7_db_number` | عددی | در همان شرط: `parameter[6:8]` big-endian. |
| `s7_area_code` | عددی | در همان شرط: `parameter[8]`. |
| `s7_return_code` | — | هنوز پیاده‌سازی نشده؛ NaN schema placeholder است و نباید انتخاب شود. |
| `s7_response_matched` | Boolean/عددی | 1 اگر `(capture,flow_id,s7_pdu_reference)` دست‌کم دو direction یکتا داشته باشد. |

## پیش‌پردازش دقیق featureهای انتخاب‌شده

تنظیم پیش‌فرض در `config/default.yaml`: `min_non_null_ratio=0.05`، `high_cardinality_max_categories=50`، `numeric_scaler=robust` و `numeric_clip=12.0`.

1. **انتخاب:** تنها featureهای profile که برای protocolهای حاضر معتبر و واقعاً در جدول هستند نگه‌داری می‌شوند؛ سپس `protocol` در صورت وجود افزوده می‌شود.
2. **کم‌پوشش:** هر ستون با نسبت non-null کمتر از 5٪ حذف و در `dropped_missing_features` manifest ثبت می‌شود. featureهای SSH/S7 که هنوز NaN هستند، این‌جا حذف می‌شوند.
3. **ثابت:** ستون با `nunique(dropna=True) <= 1` حذف و در `dropped_constant_features` ثبت می‌شود.
4. **متنیِ پرمقدار:** اگر cardinality یک ستون غیرعددی بیش از 50 باشد، فقط 50 مقدار پرتکرار نگه‌داری و بقیه به `__OTHER__` تبدیل می‌شوند. این واژگان در `categorical_value_vocabulary` ذخیره می‌شود.
5. **نوع‌دهی:** dtype عددی یا Boolean → numeric؛ بقیه → categorical. تصمیم از dtype واقعی می‌آید، نه اسم feature.

### numeric: Imputation → Robust scaling → Clip

Transformer فقط روی partition fit ساخته می‌شود؛ در Stage 1 این partition آموزش benign است تا leakage رخ ندهد.

1. مقدار NaN با median همان feature در دادهٔ fit پر می‌شود (`SimpleImputer(median)`).
2. `SafeRobustScaler` با `center=median` و `scale=Q75-Q25` دادهٔ fit اعمال می‌شود: `(x-center)/scale`.
3. اگر IQR کمتر/برابر `1e-12` باشد، standard deviation جایگزین scale می‌شود؛ اگر آن هم تقریباً صفر باشد scale=1 است.
4. خروجی به `[-12,+12]` clip می‌شود تا outlier خراب MSE LSTM-AE را غالب نکند.

اگر `numeric_scaler` از `robust` تغییر کند، `StandardScaler` جایگزین می‌شود.

### categorical: Imputation → One-Hot

1. NaN با پرتکرارترین category در دادهٔ fit پر می‌شود.
2. `OneHotEncoder(handle_unknown='ignore')` برای هر category آموزش‌دیده ستون 0/1 می‌سازد.
3. در inference، مقدار جدید در ستونِ محدودشده ابتدا `__OTHER__` می‌شود؛ در ستونِ محدودنشده توسط encoder ignore می‌شود و تمام one-hotهای همان feature صفر می‌مانند.

بنابراین یک feature متنی ممکن است چند input column مدل بسازد. ترتیب نهایی با `transformer.get_feature_names_out()` ثابت و همراه pipeline ذخیره می‌شود؛ نام‌ها یا ترتیب را دستی عوض نکنید.

## قرارداد inference و فایل‌های audit

- `*.pipeline.joblib` شامل imputer/scaler/encoder fit‌شده و `*.manifest.json` شامل featureهای انتخابی/حذف‌شده، واژگان، clip و ترتیب قرارداد است.
- در inference، feature انتخاب‌شده‌ای که در raw table نباشد به‌شکل NaN اضافه می‌شود تا imputer آموزش‌دیده آن را مدیریت کند؛ بردار ورودی مدل همیشه همان ترتیب آموزش را دارد.
- تغییر profile یعنی تغییر قرارداد ورودی؛ برای profile جدید باید model جدید آموزش داده شود.

| فایل | کاربرد |
|---|---|
| `records.parquet` | packet metadata و featureهای خام/NaN، جدا برای هر protocol و capture. |
| `records.manifest.json` | PCAP منبع، شمار packet/flow، schema و تنظیم extractor. |
| `reports/features/*-overview.parquet` | availability، صفرها، mean، median، IQR و صدک‌ها برای dashboard/API. |
| `reports/features/*-quality.parquet` | قابل استفاده بودن feature و دلیل رد/قبول. |
| `stage1/prepared-normal.parquet` | ماتریس transformed و metadata برای audit. |
| `stage1/prepared-normal.manifest.json` | قرارداد preprocessing و ترتیب inputها. |
| `stage1/prepared-normal.pipeline.joblib` | transformer frozen که هر سیستم دیگر باید در inference همان مدل مصرف کند. |
| `reports/preprocessing-comparison.parquet` | آمار قبل/بعد برای featureهای profile. |

برای تفسیر خروجی anomaly، `stage1/feature-evidence.parquet` سه feature با فاصلهٔ robust بیشتر از baseline نرمال را برای هر ردیف ثبت می‌کند. این «شاهد آماری قابل‌ردیابی» است، نه ادعای علت قطعی حمله؛ آن را کنار packet metadata، score و زمینهٔ شبکه تحلیل کنید.
