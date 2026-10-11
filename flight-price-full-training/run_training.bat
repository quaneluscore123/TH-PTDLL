@echo off
chcp 65001 >nul
set PYTHONUTF8=1
echo =====================================================================
echo   KHỞI CHẠY PIPELINE SILVER -^> GOLD -^> MACHINE LEARNING
echo   Dữ liệu gốc: F:\PKA\Code\BD\itineraries.csv (31,09 GB)
echo   Cấu hình: Intel Core i5-14500 (20 luồng) ^| RAM 32 GB (Heap 24 GB)
echo =====================================================================
echo.
echo Lựa chọn chế độ thực thi:
echo   [1] Chạy trọn gói từ Bronze -^> Silver -^> Gold -^> Huấn luyện ML (Khuyến nghị)
echo   [2] Chỉ chạy huấn luyện ML trên 100%% dữ liệu (Yêu cầu Silver đã có)
echo   [3] Chỉ chạy huấn luyện ML trên 50%% dữ liệu (~41 triệu dòng)
echo.
set /p opt="Nhập lựa chọn của bạn [1/2/3] (Mặc định: 1): "

if "%opt%"=="2" (
    echo Đang chạy huấn luyện ML trên 100%% dữ liệu...
    python train_full.py --fraction 1.0
) else if "%opt%"=="3" (
    echo Đang chạy huấn luyện ML trên 50%% dữ liệu...
    python train_full.py --fraction 0.50
) else (
    echo Đang chạy trọn gói Bronze -^> Silver -^> Gold -^> Machine Learning...
    python pipeline_silver_to_gold_ml.py
)

echo.
echo =====================================================================
echo   HOÀN THÀNH. Kết quả đã lưu tại F:\flight-data\exports_full_32gb\
echo =====================================================================
pause
