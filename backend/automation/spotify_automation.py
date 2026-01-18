"""
Spotify Selenium Automation
Fully automated Spotify playback using Selenium WebDriver
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

class SpotifyAutomation:
    def __init__(self):
        self.driver = None
        
    def play_song(self, song_name: str, artist_name: str = ""):
        """
        Fully automated Spotify playback - NO manual clicking needed!
        Opens Spotify web player, searches, and auto-clicks play button.
        """
        try:
            # Build search query
            query = f"{song_name} {artist_name}".strip().replace(" ", "+")
            url = f"https://open.spotify.com/search/{query}"
            
            # Setup Brave browser with options
            options = webdriver.ChromeOptions()
            options.binary_location = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
            
            # Use a separate profile for automation (avoids conflicts with open Brave)
            import os
            user_data_dir = os.path.expanduser(r"~\AppData\Local\BraveSoftware\Brave-Browser\User Data Selenium")
            options.add_argument(f"user-data-dir={user_data_dir}")
            
            # Add options to prevent crashes
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--start-maximized")
            options.add_argument("--disable-blink-features=AutomationControlled")
            
            # Initialize driver with Brave
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            
            logger.info(f"Opening Spotify search for: {song_name} {artist_name}")
            self.driver.get(url)
            
            # Wait for page to load and find play button
            wait = WebDriverWait(self.driver, 10)
            
            # Try multiple selectors for play button
            play_button_selectors = [
                "button[aria-label*='Play']",
                "button[data-testid='play-button']",
                "button.playButton",
                "button[title*='Play']",
                "[aria-label*='play' i]"
            ]
            
            for selector in play_button_selectors:
                try:
                    play_button = wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    time.sleep(1)  # Brief pause for stability
                    play_button.click()
                    logger.info(f"✅ Successfully clicked play button for: {song_name}")
                    return True, f"Now playing: {song_name} {artist_name}"
                except:
                    continue
            
            # If no button found, just return success (page is open)
            logger.warning("Could not find play button, but page is open")
            return True, f"Opened Spotify search for: {song_name}"
            
        except Exception as e:
            logger.error(f"Spotify automation failed: {e}")
            if self.driver:
                self.driver.quit()
            return False, f"Failed to play: {str(e)}"
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            self.driver = None
