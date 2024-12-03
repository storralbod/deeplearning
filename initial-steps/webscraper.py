from urllib.request import urlopen
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
import time
import os
from datetime import datetime, timedelta


url_da = r'https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename=marginalpdbc'
url_id = r'https://www.omie.es/es/file-download?parents%5B0%5D=precios_pibcic&filename=precios_pibcic'
download_dir = "./raw_data"

exceptions = []
years = [2023, 2024]

chrome_options = webdriver.ChromeOptions()
prefs = {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True,
    "profile.default_content_settings.popups": 0,
    "profile.default_content_setting_values.automatic_downloads": 1
}

headless = False
if headless:
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")                                                                                                                                       
chrome_options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(options=chrome_options)
try:
    for file_type, base_url in [('DA', url_da), ('ID', url_id)]:
        for year in years:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year + 1, 1, 1)
            days = (end_date - start_date).days
            for i in range(days):
                start_date = datetime(2023, 1, 1)
                date = (start_date + timedelta(days=i)).strftime('%Y%m%d')
                file_url = f"{base_url}_{date}.1"
                before_download = set(os.listdir(download_dir))
                try:
                    driver.get(file_url)
                except WebDriverException as e:
                    print(f"Error navigating to {file_url}: {e}")
                    exceptions.append(file_url)
                    continue
                download_complete = False
                for _ in range(5):
                    time.sleep(6)
                    after_download = set(os.listdir(download_dir))
                    new_files = after_download - before_download
                    if new_files:
                        download_complete = True
                        downloaded_file = new_files.pop()
                        break
                
                if not download_complete:
                    print(f"Download for {file_url} timed out.")
                    continue

                old_path = os.path.join(download_dir, downloaded_file)
                new_path = os.path.join(download_dir, f"{downloaded_file}.csv")
                os.rename(old_path, new_path)
                print(f"Downloaded and renamed file to: {new_path}")
finally:
    driver.quit()

print("Download process completed.")
exceptions and print("The following URLs failed to download:")
try:
    for y in years:
        base_url = "https://www.mibgas.es/en/file-access/MIBGAS_Data_" + str(y) + ".csv?path=AGNO_" + str(y) + "/XLS"
        before_download = set(os.listdir(download_dir))
        try:
            driver.get(base_url)
        except WebDriverException as e:
            print(f"Error navigating to {base_url}: {e}")
            exceptions.append(base_url)
            continue
        download_complete = False
        for _ in range(5):
            time.sleep(6)
            after_download = set(os.listdir(download_dir))
            new_files = after_download - before_download
            if new_files:
                download_complete = True
                downloaded_file = new_files.pop()
                break
        
        if not download_complete:
            print(f"Download for {base_url} timed out.")
            continue
        print(f"Downloaded file to: {download_dir}")
finally:
    driver.quit()


