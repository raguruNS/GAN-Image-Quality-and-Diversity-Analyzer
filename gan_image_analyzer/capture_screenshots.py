"""
Capture high-quality UI Screenshots of the running Streamlit application.
Waits for Streamlit websocket and DOM elements to fully render.
"""

import time
from playwright.sync_api import sync_playwright

def capture():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1050})
        page = context.new_page()

        print("Navigating to Streamlit app...")
        page.goto("http://localhost:8501", timeout=60000)
        
        # Wait for Streamlit app container and title
        page.wait_for_selector("h1", timeout=30000)
        time.sleep(3)
        page.screenshot(path="screenshot_landing.png")
        print("Captured screenshot_landing.png")

        # Click the primary action button to run GAN inference
        run_btn = page.locator("button:has-text('Run GAN Inference')")
        if run_btn.count() > 0:
            print("Triggering GAN inference...")
            run_btn.first.click()
            # Wait for generation to finish (wait for gallery images to load)
            page.wait_for_selector("img", timeout=45000)
            time.sleep(4)
            page.screenshot(path="screenshot_gallery.png")
            print("Captured screenshot_gallery.png")

            # Click Quality Metrics tab using get_by_role
            quality_tab = page.get_by_role("tab", name="Quality Metrics")
            if quality_tab.count() > 0:
                quality_tab.first.click()
                time.sleep(3)
                page.screenshot(path="screenshot_quality.png")
                print("Captured screenshot_quality.png")

            # Click Diversity Analysis tab
            div_tab = page.get_by_role("tab", name="Diversity Analysis")
            if div_tab.count() > 0:
                div_tab.first.click()
                time.sleep(3)
                page.screenshot(path="screenshot_diversity.png")
                print("Captured screenshot_diversity.png")

            # Click Reports & Export tab
            rep_tab = page.get_by_role("tab", name="Reports & Export")
            if rep_tab.count() > 0:
                rep_tab.first.click()
                time.sleep(3)
                page.screenshot(path="screenshot_reports.png")
                print("Captured screenshot_reports.png")

        browser.close()
        print("All screenshots successfully captured!")

if __name__ == "__main__":
    capture()
