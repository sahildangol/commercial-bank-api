import json
import time
import re
import os
import sys
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

class UniversalNepseScraper:
    def __init__(self, end_date=None):
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
       
        self.start_date = (datetime.strptime(self.end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        
        opts = Options()
    
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1920,1080")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=opts)
        self.wait = WebDriverWait(self.driver, 25)  
    
    def setup_search(self):

        
        self.driver.get("https://www.sharesansar.com/index-history-data")
        time.sleep(12)  # ✅ Longer initial load
        # Select NEPSE Index (value='12')
        try:
            index_select = self.wait.until(EC.element_to_be_clickable((By.ID, "index")))
            self.driver.execute_script("arguments[0].click();", index_select)
            time.sleep(2)
            
            # ✅ Direct select NEPSE Index
            nepse_option = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//select[@id='index']/option[contains(text(), 'NEPSE Index')]")))
            self.driver.execute_script("arguments[0].selected = true;", nepse_option)
            self.driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", nepse_option)
            print("✅ NEPSE Index selected")
        except Exception as e:
            print(f"Index select: {e}")
            # Fallback: Direct JS
            self.driver.execute_script("document.getElementById('index').value = '12';")
            self.driver.execute_script("document.getElementById('index').dispatchEvent(new Event('change'));")
            print("✅ JS NEPSE Index fallback")

        # Set dates
        from_date = self.wait.until(EC.presence_of_element_located((By.ID, "fromDate")))
        from_date.clear()
        from_date.send_keys(self.start_date)
        
        to_date = self.driver.find_element(By.ID, "toDate")
        to_date.clear()
        to_date.send_keys(self.end_date)
        time.sleep(2)
        
        # Search
        search_btn = self.wait.until(EC.element_to_be_clickable((By.ID, "btn_indxhis_submit")))
        self.driver.execute_script("arguments[0].click();", search_btn)
        time.sleep(15)  # ✅ Wait for results
        print("✅ Search completed")
    
    def parse_page(self, soup):
        """Parse ALL records from current page"""
        data = []
        table = soup.find('table', {'id': 'myTable'})
        
        if table:
            # ✅ Try multiple table body selectors
            tbody = table.find('tbody') or table.find('tbody', class_='dataTables_scrollBody')
            rows = tbody.find_all('tr') if tbody else table.find_all('tr')[1:]
            
            print(f"🔍 Found {len(rows)} rows on page")
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 9:
                    try:
                        sn_text = cols[0].get_text(strip=True)
                        date_text = cols[8].get_text(strip=True)
                        close_text = cols[4].get_text(strip=True).replace(',', '').strip()
                        turnover_text = cols[7].get_text(strip=True).replace(',', '').strip()
                        
                        # ✅ More robust validation
                        if (sn_text.isdigit() and 
                            date_text and 
                            close_text.replace('.', '').isdigit() and 
                            turnover_text):
                            
                            record = {
                                "date": date_text.strip(),
                                "id": int(sn_text),
                                "close": round(float(close_text), 2),
                                "turnover": round(float(re.sub(r'[^\d.]', '', turnover_text)), 2)
                            }
                            data.append(record)
                    except Exception as e:
                        print(f"Parse error: {e}")
                        continue
        
        return data
    
    def scrape_120_records(self):
        """✅ GUARANTEE 120 records with extended pagination"""
        self.setup_search()
        all_data = []
        page = 0
        max_pages = 100  # ✅ Increased max pages
        
        print("🚀 Scraping EXACTLY 120 records...")
        
        while len(all_data) < 120 and page < max_pages:
            print(f"📄 Page {page+1} (Total: {len(all_data)}/120)...")
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            page_data = self.parse_page(soup)
            
            # Add only NEW records (no duplicates by date)
            seen_dates = {r['date'] for r in all_data}
            new_count = 0
            
            for record in page_data:
                if record['date'] not in seen_dates and len(all_data) < 120:
                    all_data.append(record)
                    seen_dates.add(record['date'])
                    new_count += 1
            
            print(f"   ➕ Added {new_count} new records")
            
            if len(all_data) >= 120:
                print("✅ REACHED 120 records!")
                break
            
            # Next page - Multiple strategies
            try:
                # Strategy 1: DataTables next button
                next_btn = self.driver.find_element(By.ID, "myTable_next")
                if "disabled" not in next_btn.get_attribute("class") and "current" not in next_btn.get_attribute("class"):
                    self.driver.execute_script("arguments[0].click();", next_btn)
                    time.sleep(6)
                    page += 1
                    continue
            except:
                pass
            
            try:
                # Strategy 2: Pagination links
                page_links = self.driver.find_elements(By.CSS_SELECTOR, "#myTable_paginate a")
                if page_links:
                    next_link = page_links[-1] if "Next" in page_links[-1].text else page_links[page % len(page_links)]
                    if next_link.is_enabled():
                        self.driver.execute_script("arguments[0].click();", next_link)
                        time.sleep(6)
                        page += 1
                        continue
            except:
                pass
            
            try:
                # Strategy 3: Direct page number
                page_input = self.driver.find_element(By.ID, "myTable_length")
                page_num = str(page + 2)
                self.driver.execute_script(f"document.querySelector('#myTable_paginate a.page-link').click();")
                time.sleep(6)
                page += 1
            except:
                break
        
        self.driver.quit()
        final_data = sorted(all_data[:120], key=lambda x: x['date'], reverse=True)
        print(f"✅ Final: {len(final_data)} records collected")
        return final_data
    
    def save_data(self, data):
        """Save EXACTLY 120 records"""
        # Load existing data
        existing = []
        if os.path.exists('nepse_index.json'):
            try:
                with open('nepse_index.json', 'r') as f:
                    existing = json.load(f)
            except:
                pass
        
        # Combine and deduplicate by date
        all_records = existing + data
        seen_dates = {}
        unique_records = []
        
        for record in sorted(all_records, key=lambda x: x['date'], reverse=True):
            if record['date'] not in seen_dates:
                unique_records.append(record)
                seen_dates[record['date']] = True
                if len(unique_records) >= 120:
                    break
        
        # Save exactly 120 newest records
        final_data = unique_records[:120]
        
        with open('nepse_index.json', 'w') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved EXACTLY {len(final_data)} records to nepse_index.json")
        return final_data

def get_end_date():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return datetime.now().strftime('%Y-%m-%d')

def main():
    print("🚀 NEPSE INDEX SCRAPER - EXACTLY 120 RECORDS")
    end_date = get_end_date()
    print(f"📅 Target end date: {end_date}")
    
    scraper = UniversalNepseScraper(end_date)
    
    try:
        data = scraper.scrape_120_records()
        
        if data and len(data) >= 120:
            final_data = scraper.save_data(data)
            print(f"\n🎉 ✅ SUCCESS! EXACTLY {len(final_data)} records scraped!")
            
            print("\n📊 Latest 5 records:")
            for record in final_data[:5]:
                print(f"   📈 {record['date']}: Close={record['close']} | Turnover={record['turnover']:,.0f}")
            
            print(f"\n💾 nepse_index.json → {len(final_data)} PERFECT records")
        else:
            print(f"❌ Only {len(data)} records found. Try earlier end_date.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    finally:
        try:
            scraper.driver.quit()
        except:
            pass

if __name__ == "__main__":
    main()
