"""Cấu hình dùng chung cho mọi notebook: đường dẫn dữ liệu và khởi tạo Spark trên Windows."""
import os
import sys

# --- Đường dẫn ---
CSV_PATH = "D:/TH-PTDLL/data/flightprices/itineraries.csv"   # file gốc 31 GB, 82 triệu dòng
DATA_DIR = "E:/flight-data"                                    # Bronze/Silver/Gold đặt ở ổ E
BRONZE = f"{DATA_DIR}/bronze/itineraries"
SILVER = f"{DATA_DIR}/silver/itineraries"
GOLD = f"{DATA_DIR}/gold"
EXTERNAL = f"{DATA_DIR}/external"       # dữ liệu nguồn ngoài (sân bay, thời tiết, CPI, DOT...)
EXPORTS = f"{DATA_DIR}/exports"         # CSV kết quả để nộp / dùng cho dashboard
SPARK_TMP = f"{DATA_DIR}/spark-tmp"
HADOOP_HOME = "E:/hadoop"

ML_FRACTION = 0.05      # tỉ lệ mẫu cho huấn luyện mô hình (5%; máy 16 GB RAM có thể tăng lên 0.10)
SEED = 42

for d in (EXTERNAL, EXPORTS, SPARK_TMP, GOLD):
    os.makedirs(d, exist_ok=True)


def get_spark(app_name="flight-prices", driver_memory="9g"):
    """Tạo SparkSession chạy local, dùng hết CPU, file tạm đặt ở ổ E."""
    os.environ.setdefault("HADOOP_HOME", HADOOP_HOME)
    os.environ["PATH"] = f"{HADOOP_HOME}/bin;" + os.environ["PATH"]
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.local.dir", os.path.normpath(SPARK_TMP))
        .config("spark.sql.shuffle.partitions", "96")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
