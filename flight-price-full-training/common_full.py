"""Cấu hình PySpark tối ưu cho máy 32 GB RAM và CPU Intel Core i5-14500 (20 luồng), tự động nhận diện cấu hình máy thực tế."""
import os
import sys
import glob
import shutil

# --- 1. Tự động phát hiện JAVA_HOME (ưu tiên biến môi trường hiện tại, sau đó quét C:\Program Files\Java) ---
def _find_java_home():
    env_java = os.environ.get("JAVA_HOME")
    if env_java and os.path.exists(os.path.join(env_java, "bin", "java.exe")):
        return env_java
    candidates = sorted(glob.glob(r"C:\Program Files\Java\jdk*"), reverse=True)
    for c in candidates:
        if os.path.exists(os.path.join(c, "bin", "java.exe")):
            return c
    return env_java

JAVA_HOME = _find_java_home()
if JAVA_HOME:
    os.environ["JAVA_HOME"] = JAVA_HOME

# --- 2. Tự động phát hiện HADOOP_HOME (F:/PKA/Code/DLL/hadoop) ---
CANDIDATE_HADOOP = [
    os.environ.get("HADOOP_HOME"),
    "F:/PKA/Code/DLL/hadoop",
    r"F:\PKA\Code\DLL\hadoop",
    "C:/hadoop",
]
HADOOP_HOME = next((p for p in CANDIDATE_HADOOP if p and os.path.exists(os.path.join(p, "bin", "winutils.exe"))), "F:/PKA/Code/DLL/hadoop")
os.environ["HADOOP_HOME"] = HADOOP_HOME

# Bổ sung System32 vào PATH để các tiện ích hệ thống (taskkill) hoạt động bình thường
sys32 = r"C:\Windows\system32;C:\Windows;"
os.environ["PATH"] = f"{HADOOP_HOME}/bin;{sys32}" + os.environ.get("PATH", "")

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
    kh_matches = glob.glob(os.path.join(user_home, r".cache\kagglehub\**\itineraries.csv"), recursive=True)
    if kh_matches:
        CSV_PATH = kh_matches[0]

# --- 4. Thư mục lưu trữ chính (Ưu tiên ổ F: có 95 GB trống) ---
def _pick_best_data_dir():
    candidates = ["C:/flight-data", "F:/flight-data", "D:/flight-data"]
    # 1. Ưu tiên thư mục đã có sẵn dữ liệu tầng Silver đã tiền xử lý
    for c in candidates:
        if os.path.exists(f"{c}/silver/itineraries"):
            return c
    # 2. Nếu chưa có, chọn ổ đĩa có dung lượng trống lớn nhất (> 10GB)
    best_dir = "C:/flight-data"
    best_free = 0
    for c in candidates:
        drive = os.path.splitdrive(c)[0] + "/"
        try:
            free = shutil.disk_usage(drive).free
            if free > best_free and free > 10 * 1024**3:
                best_free = free
                best_dir = c
        except Exception:
            continue
    return best_dir

DATA_DIR = os.environ.get("FLIGHT_DATA_DIR", _pick_best_data_dir())
BRONZE = f"{DATA_DIR}/bronze/itineraries"
SILVER = f"{DATA_DIR}/silver/itineraries"
GOLD = f"{DATA_DIR}/gold"
EXPORTS_FULL = f"{DATA_DIR}/exports_full_32gb"
MODELS_FULL = f"{DATA_DIR}/models_full_32gb"
SPARK_TMP_FULL = f"{DATA_DIR}/spark-tmp-32gb"
CHECKPOINT_DIR = f"{DATA_DIR}/spark-tmp-32gb/checkpoints"

for d in (EXPORTS_FULL, MODELS_FULL, SPARK_TMP_FULL, CHECKPOINT_DIR):
    os.makedirs(d, exist_ok=True)

TRAIN_FRACTION = float(os.environ.get("FULL_TRAIN_FRACTION", "1.0"))
SEED = 42


def get_spark_32gb(app_name="flight-price-full-32gb", driver_memory="20g"):
    """Khởi tạo SparkSession tối ưu hóa tối đa cho 32 GB RAM (20 GB JVM Heap + 12 GB cho LightGBM/OS) và i5-14500 (20 luồng)."""
    os.environ.setdefault("HADOOP_HOME", HADOOP_HOME)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    java_opts = (
        "-XX:+UseG1GC "
        "-XX:InitiatingHeapOccupancyPercent=35 "
        "-XX:G1ReservePercent=15 "
        "-XX:MaxGCPauseMillis=200 "
        "-XX:+ParallelRefProcEnabled"
    )

    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[20]")  # Tận dụng toàn bộ 20 luồng của Intel Core i5-14500
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "6g")
        .config("spark.local.dir", os.path.normpath(SPARK_TMP_FULL))
        .config("spark.sql.shuffle.partitions", "160")
        .config("spark.default.parallelism", "40")
        .config("spark.memory.fraction", "0.80")
        .config("spark.memory.storageFraction", "0.30")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.extraJavaOptions", java_opts)
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    spark.sparkContext.setCheckpointDir(os.path.normpath(CHECKPOINT_DIR))
    return spark
