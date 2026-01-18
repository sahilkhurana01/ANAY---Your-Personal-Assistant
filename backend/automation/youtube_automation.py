"""
YouTube Selenium Automation
Fully automated YouTube playback using Selenium WebDriver
"""
import logging
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

class YouTubeAutomation:
    def __init__(self):
        self.driver = None
        
    def play_video(self, video_name: str):
        """
        Fully automated YouTube playback - connects to your open Brave browser!
        """
        try:
            # Build search query
            query = video_name.strip().replace(" ", "+")
            url = f"https://www.youtube.com/results?search_query={query}"
            
            # Setup Chrome/Brave options to connect to existing browser
            options = webdriver.ChromeOptions()
            options.binary_location = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
            
            # Connect to existing browser via remote debugging
            # User needs to start Brave with: --remote-debugging-port=9222
            try:
                options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            except:
                # Fallback: Just open URL in default browser
                import webbrowser
                webbrowser.open(url)
                return True, f"Opened YouTube for: {video_name} (Please enable remote debugging for full automation)"
            
            logger.info(f"Opening YouTube search for: {video_name}")
            self.driver.get(url)
            
            # Wait for page to load and find first video
            wait = WebDriverWait(self.driver, 10)
            
            # Try to find and click first video thumbnail
            video_selectors = [
                "a#video-title",
                "ytd-video-renderer a#thumbnail",
                "a.yt-simple-endpoint.ytd-video-renderer"
            ]
            
            for selector in video_selectors:
                try:
                    video_link = wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    time.sleep(1)  # Brief pause for stability
                    video_link.click()
                    logger.info(f"✅ Successfully clicked video for: {video_name}")
                    return True, f"Now playing: {video_name}"
                except:
                    continue
            
            # If no video found, just return success (page is open)
            logger.warning("Could not find video link, but page is open")
            return True, f"Opened YouTube search for: {video_name}"
            
        except Exception as e:
            logger.error(f"YouTube automation failed: {e}")
            if self.driver:
                self.driver.quit()
            return False, f"Failed to play: {str(e)}"
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            self.driver = None
