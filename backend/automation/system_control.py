"""
System Control Module (Refactored)
Focuses on OS-level process management, app launching, and system info.
"""
import logging
import psutil
import subprocess
import os
import platform
from typing import List, Dict
from automation.context_manager import ContextManager

logger = logging.getLogger(__name__)

class SystemControl:
    """Controls OS processes and applications."""
    
    def __init__(self):
        self.os_type = platform.system()
        self.ctx_mgr = ContextManager()
        
    def get_system_stats(self) -> Dict:
        """Get CPU/RAM usage."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "battery": self._get_battery(),
            "os": self.os_type
        }

    def _get_battery(self):
        try:
            battery = psutil.sensors_battery()
            return {"percent": battery.percent, "plugged": battery.power_plugged} if battery else "Unknown"
        except: return "No Battery"

    def launch_app(self, app_name: str):
        """Intelligent application launcher."""
        try:
            # Common paths map (Simplified)
            common_apps = {
                "calculator": "calc.exe",
                "notepad": "notepad.exe",
                "chrome": "chrome.exe",
                "browser": self._get_default_browser(),  # Dynamic browser detection
                "explorer": "explorer.exe",
                "cmd": "cmd.exe",
                "spotify": "spotify.exe",
                "code": "code", # VS Code
                "vscode": "code",
                "whatsapp": os.path.expanduser("~/AppData/Local/WhatsApp/WhatsApp.exe"),
                "cursor": os.path.expanduser("~/AppData/Local/Programs/cursor/Cursor.exe"),
                "comet": "comet.exe" # Assuming in path
            }
            
            cmd = common_apps.get(app_name.lower(), app_name)
            
            # Special handling for VS Code if not in path
            if app_name.lower() in ["code", "vs code", "vscode"]:
                cmd = "code"
                
            # Special handling for Cursor
            if app_name.lower() == "cursor":
                cursor_path = os.path.expanduser("~/AppData/Local/Programs/cursor/Cursor.exe")
                if os.path.exists(cursor_path):
                    cmd = cursor_path
                else:
                    cmd = "Cursor.exe" # Hope it's in path
            
            # Special handling for Spotify - try multiple paths
            if app_name.lower() == "spotify":
                spotify_paths = [
                    os.path.expanduser("~/AppData/Roaming/Spotify/Spotify.exe"),
                    "C:\\Users\\%USERNAME%\\AppData\\Roaming\\Spotify\\Spotify.exe",
                    "spotify.exe"  # Try from PATH
                ]
                spotify_found = False
                for path in spotify_paths:
                    expanded_path = os.path.expandvars(path)
                    if os.path.exists(expanded_path):
                        cmd = expanded_path
                        spotify_found = True
                        logger.info(f"Found Spotify at: {expanded_path}")
                        break
                
                if not spotify_found:
                    logger.warning("Spotify not found in common locations, trying spotify.exe from PATH")
                    cmd = "spotify.exe"
            
            logger.info(f"Attempting to launch: {cmd}")
            
            if self.os_type == "Windows":
                 result = subprocess.Popen(cmd, shell=True)
                 logger.info(f"Launched process PID: {result.pid}")
            elif self.os_type == "Darwin": # MacOS
                 subprocess.Popen(["open", "-a", app_name])
            else:
                 subprocess.Popen([app_name])
                 
            self.ctx_mgr.update_context({"last_opened_app": app_name})
            return True, f"Launched {app_name}"
        except Exception as e:
            logger.error(f"Failed to launch {app_name}: {e}")
            return False, f"Failed to launch: {e}"
    
    def _get_default_browser(self):
        """Get the default browser executable name."""
        try:
            import winreg
            # Read default browser from Windows registry
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice") as key:
                prog_id = winreg.QueryValueEx(key, "ProgId")[0]
                
            # Map ProgId to executable
            browser_map = {
                "ChromeHTML": "chrome.exe",
                "BraveHTML": "brave.exe",
                "FirefoxURL": "firefox.exe",
                "MSEdgeHTM": "msedge.exe"
            }
            return browser_map.get(prog_id, "chrome.exe")  # Default to Chrome
        except:
            return "chrome.exe"  # Fallback
    
    def open_url(self, url: str):
        """Open a URL in the default browser - INSTANT execution!"""
        try:
            import webbrowser
            webbrowser.open(url)
            logger.info(f"Opened URL: {url}")
            return True, f"Opened {url}"
        except Exception as e:
            logger.error(f"Failed to open URL {url}: {e}")
            return False, f"Failed to open URL: {e}"
    
    def play_spotify_song(self, song_name: str, artist_name: str = ""):
        """INSTANT Spotify playback - opens in your existing browser!"""
        try:
            import webbrowser
            # Build search query
            query = f"{song_name} {artist_name}".strip().replace(" ", "+")
            url = f"https://open.spotify.com/search/{query}"
            
            logger.info(f"Opening Spotify search for: {song_name} {artist_name}")
            webbrowser.open(url)
            
            return True, f"Now playing: {song_name} {artist_name}"
        except Exception as e:
            logger.error(f"Spotify opening failed: {e}")
            return False, f"Failed to play: {str(e)}"
    
    def play_youtube_video(self, video_name: str):
        """Fully automated YouTube playback with Selenium!"""
        try:
            from automation.youtube_automation import YouTubeAutomation
            youtube = YouTubeAutomation()
            success, message = youtube.play_video(video_name)
            logger.info(f"YouTube automation: {message}")
            return success, message
        except Exception as e:
            logger.error(f"YouTube automation failed: {e}")
            # Fallback to simple URL opening
            import webbrowser
            url = f"https://www.youtube.com/results?search_query={video_name.replace(' ', '+')}"
            webbrowser.open(url)
            return True, f"Opened YouTube for: {video_name}"

    def close_app(self, app_name: str):
        """Kill a process by name."""
        killed_count = 0
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if app_name.lower() in proc.info['name'].lower():
                    proc.kill()
                    killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if killed_count > 0:
            return True, f"Closed {killed_count} instances of {app_name}"
        return False, f"No running process found for {app_name}"

    def shutdown(self):
        if self.os_type == "Windows":
            os.system("shutdown /s /t 10")
        else:
            os.system("shutdown now")
        return True, "Initiating shutdown in 10s"
