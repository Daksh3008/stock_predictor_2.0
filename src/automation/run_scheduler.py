# src/automation/run_scheduler.py
import os, sys, time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import schedule
from datetime import datetime

def run_predict():
    print(f"[{datetime.now()}] Running predict_daily.py")
    os.system("python src/automation/predict_daily.py")

def run_validate():
    print(f"[{datetime.now()}] Running validate_close.py")
    os.system("python src/automation/validate_close.py")

# Schedule Mon-Fri
for day in ["monday","tuesday","wednesday","thursday","friday"]:
    schedule.every().__getattribute__(day).at("11:00").do(run_predict)
    schedule.every().__getattribute__(day).at("16:00").do(run_validate)

print("⏳ Scheduler started... (Ctrl+C to stop)")
while True:
    schedule.run_pending()
    time.sleep(30)
