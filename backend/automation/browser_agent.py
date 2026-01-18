"""
Browser Agent Module
Uses Selenium for full browser automation if available, otherwise falls back to basic webbrowser control.
"""
import logging
import time
import os
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class BrowserAgent:
    """Controls web browsers for navigation and interaction."""
    
    def __init__(self):
        self.driver = None
        self.mode = "basic" # basic (webbrowser module) or advanced (selenium)
        self.is_setup = False
        
        # Check if selenium is available
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            self.mode = "advanced"
            self.webdriver = webdriver
            self.Service = Service
            self.ChromeDriverManager = ChromeDriverManager
        except ImportError:
            logger.warning("Selenium not found. Using basic browser control.")
            self.mode = "basic"

    def _init_driver(self):
        """Lazy initialization of Chrome driver."""
        if self.driver: return True
        if self.mode != "advanced": return False
        
        try:
            options = self.webdriver.ChromeOptions()
            options.add_argument("--start-maximized")
            # Don't detach so we can control it
            options.add_experimental_option("detach", True)
            
            service = self.Service(self.ChromeDriverManager().install())
            self.driver = self.webdriver.Chrome(service=service, options=options)
            self.is_setup = True
            return True
        except Exception as e:
            logger.error(f"Failed to init Selenium: {e}")
            self.mode = "basic" # Fallback
            return False

    def open_url(self, url: str):
        """Open a website."""
        if not url.startswith('http'):
            url = 'https://' + url
            
        if self.mode == "advanced" and self._init_driver():
            try:
                self.driver.get(url)
                return True
            except Exception as e:
                logger.error(f"Selenium nav failed: {e}")
                
        # Fallback to basic
        import webbrowser
        webbrowser.open(url)
        return True

    def click_element(self, selector: str, by: str = "id"):
        """Click an item (Selenium only)."""
        if self.mode != "advanced" or not self.driver:
            return False
            
        try:
            from selenium.webdriver.common.by import By
            by_map = {
                "id": By.ID,
                "xpath": By.XPATH,
                "class": By.CLASS_NAME,
                "name": By.NAME,
                "selector": By.CSS_SELECTOR
            }
            element = self.driver.find_element(by_map.get(by, By.ID), selector)
            element.click()
            return True
        except Exception as e:
            logger.error(f"Click element failed: {e}")
            return False

    def type_into(self, selector: str, text: str, by: str = "id"):
        """Type into input field (Selenium only)."""
        if self.mode != "advanced" or not self.driver:
            return False
            
        try:
            from selenium.webdriver.common.by import By
            by_map = {
                "id": By.ID,
                "xpath": By.XPATH,
                "class": By.CLASS_NAME,
                "name": By.NAME,
                "selector": By.CSS_SELECTOR
            }
            element = self.driver.find_element(by_map.get(by, By.ID), selector)
            element.send_keys(text)
            return True
        except Exception as e:
            logger.error(f"Type into failed: {e}")
            return False

    def close(self):
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def open_youtube_and_play(self, query: str):
        """Open YouTube and play a video using keyboard automation."""
        try:
            # Open YouTube
            import webbrowser
            webbrowser.open(f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}")
            
            # Note: Actual playback requires keyboard automation from input_controller
            # This is handled by the task planner workflow
            return True, f"Opened YouTube search for: {query}"
        except Exception as e:
            logger.error(f"YouTube automation failed: {e}")
            return False, str(e)
    
    def open_spotify_web_and_play(self, query: str):
        """Open Spotify web player and search for music."""
        try:
            # Open Spotify web player
            import webbrowser
            webbrowser.open(f"https://open.spotify.com/search/{query.replace(' ', '%20')}")
            
            return True, f"Opened Spotify web player for: {query}"
        except Exception as e:
            logger.error(f"Spotify web automation failed: {e}")
            return False, str(e)
    
    def get_default_browser(self):
        """Detect the default browser on the system."""
        try:
            import webbrowser
            # Get the default browser
            browser = webbrowser.get()
            browser_name = browser.name if hasattr(browser, 'name') else "default"
            return browser_name
        except:
            return "chrome"  # Fallback to chrome
