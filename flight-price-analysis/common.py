"""Cấu hình dùng chung cho mọi notebook: tự động nhận diện môi trường thực tế của máy tính."""
import os
import sys
import glob
import shutil

# --- Tự động phát hiện JAVA_HOME (ưu tiên biến môi trường hiện tại, sau đó quét C:\Program Files\Java) ---
def _find_java_home():
    env_java = os.environ.get("JAVA_HOME")
    if env_java and os.path.exists(os.path.join(env_java, "bin", "java.exe")):
        return env_java
    # Quét các thư mục JDK cài đặt trong Program Files
    candidates = sorted(glob.glob(r"C:\Program Files\Java\jdk*"), reverse=True)
    for c in candidates:
        if os.path.exists(os.path.join(c, "bin", "java.exe")):
            return c
    return env_java

JAVA_HOME = _find_java_home()
if JAVA_HOME:
    os.environ["JAVA_HOME"] = JAVA_HOME

# --- Tự động phát hiện HADOOP_HOME (chứa winutils.exe và hadoop.dll) ---
CANDIDATE_HADOOP = [
    os.environ.get("HADOOP_HOME"),
    "F:/PKA/Code/DLL/hadoop",
    r"F:\PKA\Code\DLL\hadoop",
    "C:/hadoop",
]
HADOOP_HOME = next((p for p in CANDIDATE_HADOOP if p and os.path.exists(os.path.join(p, "bin", "winutils.exe"))), "F:/PKA/Code/DLL/hadoop")
os.environ["HADOOP_HOME"] = HADOOP_HOME

# Đảm bảo System32 luôn có trong PATH để các công cụ hệ thống như taskkill hoạt động
sys32 = r"C:\Windows\system32;C:\Windows;"
os.environ["PATH"] = f"{HADOOP_HOME}/bin;{sys32}" + os.environ.get("PATH", "")

# --- Tự động phát hiện file itineraries.csv thực tế trên máy ---
user_home = os.path.expanduser("~")
CANDIDATE_CSV = [
    os.environ.get("FLIGHT_CSV_PATH"),
    r"F:\PKA\Code\BD\itineraries.csv",
    "F:/PKA/Code/BD/itineraries.csv",
    os.path.join(user_home, r".cache\kagglehub\datasets\dilwong\flightprices\versions\1\itineraries.csv"),
    r"C:\Users\phapd\.cache\kagglehub\datasets\dilwong\flightprices\versions\1\itineraries.csv",
    "F:/flight-data/itineraries.csv",
]
CSV_PATH = next((p for p in CANDIDATE_CSV if p and os.path.exists(p)), "")
if not CSV_PATH:
    # Nếu chưa tìm thấy, quét tìm trong cache kagglehub
    kh_matches = glob.glob(os.path.join(user_home, r".cache\kagglehub\**\itineraries.csv"), recursive=True)
    if kh_matches:
        CSV_PATH = kh_matches[0]

# --- Thư mục lưu trữ chính (Ưu tiên ổ F: có dung lượng trống lớn nhất) ---
def _pick_best_data_dir():
    candidates = ["F:/flight-data", "C:/flight-data", "D:/flight-data"]
    best_dir = "F:/flight-data"
    best_free = 0
    for c in candidates:
        drive = os.path.splitdrive(c)[0] + "/"
        try:
            free = shutil.disk_usage(drive).free
            if free > best_free and free > 5 * 1024**3:  # tối thiểu 5GB
                best_free = free
                best_dir = c
        except Exception:
            continue
    return best_dir

DATA_DIR = os.environ.get("FLIGHT_DATA_DIR", _pick_best_data_dir())
BRONZE = f"{DATA_DIR}/bronze/itineraries"
SILVER = f"{DATA_DIR}/silver/itineraries"
GOLD = f"{DATA_DIR}/gold"
EXTERNAL = f"{DATA_DIR}/external"       # dữ liệu nguồn ngoài (sân bay, thời tiết, CPI, DOT...)
EXPORTS = f"{DATA_DIR}/exports"         # CSV kết quả để nộp / dùng cho dashboard
SPARK_TMP = f"{DATA_DIR}/spark-tmp"

ML_FRACTION = 0.05      # tỉ lệ mẫu cho huấn luyện mô hình (mặc định cho máy 16GB)
SEED = 42

for d in (EXTERNAL, EXPORTS, SPARK_TMP, GOLD):
    os.makedirs(d, exist_ok=True)


def get_spark(app_name="flight-prices", driver_memory="24g"):
    """Tạo SparkSession chạy local 20 luồng, 24 GB RAM, file tạm đặt ở DATA_DIR."""
    os.environ.setdefault("HADOOP_HOME", HADOOP_HOME)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[20]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "6g")
        .config("spark.local.dir", os.path.normpath(SPARK_TMP))
        .config("spark.sql.shuffle.partitions", "160")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
