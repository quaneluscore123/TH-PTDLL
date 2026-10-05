"""Tạo các file mẫu nhỏ từ itineraries.csv (31 GB) để mở được bằng Excel / pandas.

Chạy:  python flight-price-analysis/make_samples.py
"""
import os

SRC = "D:/TH-PTDLL/data/flightprices/itineraries.csv"
OUT_DIR = "D:/TH-PTDLL/data/flightprices"
HEAD_ROWS = 1_000          # 1.000 dòng đầu tiên
EVERY_N = 820              # lấy 1 dòng mỗi 820 dòng -> khoảng 100.000 dòng trải đều cả file

head_path = f"{OUT_DIR}/itineraries_first_1000.csv"
sample_path = f"{OUT_DIR}/itineraries_sample_100k.csv"

size = os.path.getsize(SRC)
read = 0
with open(SRC, encoding="utf-8", newline="") as src, \
     open(head_path, "w", encoding="utf-8", newline="") as head, \
     open(sample_path, "w", encoding="utf-8", newline="") as sample:
    header = src.readline()
    head.write(header)
    sample.write(header)
    n_sample = 0
    for i, line in enumerate(src):
        read += len(line)
        if i < HEAD_ROWS:
            head.write(line)
        if i % EVERY_N == 0:
            sample.write(line)
            n_sample += 1
        if i % 5_000_000 == 0:
            print(f"{i:>11,} dòng  ~{read / size:5.1%}", flush=True)

print(f"Tổng số dòng dữ liệu: {i + 1:,}")
print(f"Đã ghi {head_path} ({HEAD_ROWS:,} dòng)")
print(f"Đã ghi {sample_path} ({n_sample:,} dòng)")
