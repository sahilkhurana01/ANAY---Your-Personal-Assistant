"""
Task Planner Module
The "Brain" that orchestrates automation tools based on user prompts.
"""
import logging
import json
import re
from typing import Dict, Any, List, Optional
import asyncio

# Import our tools
from automation.system_control import SystemControl
from automation.file_manager import FileManager
from automation.browser_agent import BrowserAgent
from automation.input_controller import InputController
from automation.safety_guard import SafetyGuard
from automation.context_manager import ContextManager

logger = logging.getLogger(__name__)

class TaskPlanner:
    """
    Decomposes natural language requests into executable tool commands.
    Uses Persistent Context to resolve intent.
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client  # Expects GroqLLM or similar interface
        
        # Initialize Tools
        self.system = SystemControl()
        self.files = FileManager()
        self.browser = BrowserAgent()
        self.input = InputController()
        self.safety = SafetyGuard()
        self.ctx = ContextManager()
    
    async def execute_plan(self, user_prompt: str) -> str:
        """
        Main entry point:
        1. Load Context
        2. Resolve Intent (is it a knowledge query or action?)
        3. Break down step-by-step
        4. Execute
        """
        context = self.ctx.get_context()
        logger.info(f"Planning with context: {context}")
        
        # 1. Expand "it", "this" using Context Logic
        refined_prompt = self._resolve_references(user_prompt, context)
        logger.info(f"Refined Prompt: {refined_prompt}")
        
        # 2. Generate Plan using LLM
        plan = await self._generate_plan(refined_prompt, context)
        
        # 3. Validation
        if not plan or not plan.get("steps"):
            logger.info("No execution steps found. Treating as conversational/knowledge query.")
            # If the plan is empty, it means the LLM thinks it's not a PC automation task.
            # In this architecture, we return specific signal or string for the main loop to handle conversational reply.
            return "NO_ACTION_REQUIRED"

        # 4. Execution Loop
        results = []
        for step in plan.get("steps", []):
            tool = step.get("tool")
            action = step.get("action")
            params = step.get("params", {})
            
            # Safety Check
            is_safe, reason = self.safety.validate_action(tool, params)
            if not is_safe:
                logger.warning(f"Safety Block: {reason}")
                return f"I couldn't complete the task because: {reason}"
                
            # Execute
            try:
                res = self._run_tool(tool, action, params)
                
                # Intelligent Output Filtering
                # If tuple (True, "Message"), take message.
                # If tuple (True,), ignore.
                # If boolean, ignore.
                msg = ""
                if isinstance(res, tuple) and len(res) >= 2:
                    msg = str(res[1])
                elif isinstance(res, str):
                    msg = res
                
                # Only append meaningful messages
                if msg and "True" not in msg and "False" not in msg:
                    results.append(msg)
                    
            except Exception as e:
                logger.error(f"Execution Error on {action}: {e}")
                return f"Error executing step {action}: {e}"
        
        # 5. Summarize
        if not results:
            summary = "Done." # Fallback if only booleans returned
        else:
            summary = " and ".join(results)
            
        self.ctx.update_context({"last_task_summary": summary})
        
        # Return the clean summary to be spoken
        return summary

    def _resolve_references(self, text: str, ctx: Dict) -> str:
        """
        Replace pronouns with actual context paths.
        """
        lower_text = text.lower()
        
        # Priority resolution
        target_file = (
            ctx.get("last_modified_file") or 
            ctx.get("last_opened_file") or 
            ctx.get("last_created_file")
        )
        
        target_app = ctx.get("last_opened_app")
        
        if target_file and (" it" in lower_text or "that file" in lower_text or "the file" in lower_text):
            # Simple replacement strategy
            text = re.sub(r'\b(it|that file|the file)\b', f'the file "{target_file}"', text, flags=re.IGNORECASE)
        
        # Specific code fix resolution
        if target_file and ("fix the code" in lower_text or "modify the code" in lower_text):
             text = text.replace("the code", f'the code in "{target_file}"')
             
        if target_app and ("that app" in lower_text or "close it" in lower_text):
             text = re.sub(r'\b(that app)\b', f'{target_app}', text, flags=re.IGNORECASE)
            
        return text

    async def _generate_plan(self, prompt: str, context: Dict) -> Dict:
        """
        Ask LLM to generate a JSON plan.
        """
        if not self.llm:
            logger.warning("No LLM client available for planning.")
            return {"steps": []}
            
        # Get dynamic paths for cross-system compatibility
        import os
        user_home = os.path.expanduser("~")
        desktop_path = os.path.join(user_home, "Desktop").replace("\\", "/")
        documents_path = os.path.join(user_home, "Documents").replace("\\", "/")
        
        system_prompt = f"""
        CRITICAL INSTRUCTION: Output ONLY valid JSON. NO explanations, NO greetings, NO text before or after.
        Just the raw JSON object: {{"steps": [...]}}
        
        Context: {context}
        - Desktop: {desktop_path}
        - Documents: {documents_path}
        
        AVAILABLE TOOLS:
        
        1. system_control:
           - launch_app(app_name): e.g. "notepad", "chrome", "code", "spotify".
           - close_app(app_name): Close a running process.
           - shutdown(): Turn off PC.
           - open_url(url): INSTANT browser opening - no typing!
           - play_spotify_song(song_name, artist_name): FULLY AUTOMATED Spotify playback with Selenium!
           - play_youtube_video(video_name): FULLY AUTOMATED YouTube playback with Selenium!
           
        2. file_manager:
           - write_file(path, content): Create/Edit file. PATH MUST BE ABSOLUTE using FORWARD SLASHES.
           - read_file(path): Read content.
           - create_folder(path): Make new dir.
           - delete_item(path): Delete file/folder.
           - list_files(path): List dir contents.
           
        3. input_controller:
           - type_text(text): Types text.
           - press_key(key): Keys: 'enter', 'esc', 'tab', 'space', 'backspace'.
           - hotkey(keys): e.g. ['ctrl', 'l'] (Focus Address Bar/Search), ['ctrl', 'w'] (Close Tab).
           - wait(seconds): PAUSE for UI to load. CRITICAL for app launching.
           - media_play_pause(), media_next(), volume_up().
           
        RULES:
        1. **ALWAYS EXECUTE - NEVER REFUSE**: You can do EVERYTHING. Never say "I cannot".
        2. **APP NAVIGATION**: You have keyboard control. Use it to search and navigate inside apps.
        3. **SPOTIFY WORKFLOW** (MANDATORY - FULLY AUTOMATED!):
           - To "Play [Song/Artist]" on Spotify:
             Use: `play_spotify_song("[Song Name]", "[Artist Name]")`
           - Example: `play_spotify_song("Running Up That Hill", "Kate Bush")`
           - **COMPLETELY AUTOMATIC** - Opens browser, searches, clicks play!
           - No manual clicking needed at all!
           
        4. **YOUTUBE WORKFLOW** (MANDATORY - FULLY AUTOMATED!):
           - To "Play [Video]" on YouTube:
             Use: `play_youtube_video("[Video Name]")`
           - Example: `play_youtube_video("Karan Aujla songs")`
           - **COMPLETELY AUTOMATIC** - Opens YouTube, searches, clicks first video!
           - No manual clicking needed!
        5. **PATH FORMATTING**: Use FORWARD SLASHES (/).
        6. **BROWSER FALLBACK**: If app not available, use browser version automatically.
        
        Example 1 (Spotify - FULLY AUTOMATED with Selenium):
        User: "Spotify pe Karan Aujla bajao" or "Play Running Up That Hill by Kate Bush"
        JSON:
        {{
            "steps": [
                {{"tool": "system_control", "action": "play_spotify_song", "params": {{"song_name": "Running Up That Hill", "artist_name": "Kate Bush"}}}}
            ]
        }}
        
        Example 2 (YouTube - FULLY AUTOMATED with Selenium):
        User: "YouTube pe Karan Aujla ka video play karo"
        JSON:
        {{
            "steps": [
                {{"tool": "system_control", "action": "play_youtube_video", "params": {{"video_name": "Karan Aujla songs"}}}}
            ]
        }}
        
        Example 3 (File - use dynamic paths):
        User: "Create hello.txt on desktop"
        JSON:
        {{
            "steps": [
                 {{"tool": "file_manager", "action": "write_file", "params": {{"path": "{desktop_path}/hello.txt", "content": "Hello"}}}}
            ]
        }}
        
        Example 4 (Chat/Knowledge - only when NO action possible):
        User: "Tell me a joke"
        JSON:
        {{ "steps": [] }}
        """
        
        # Inject Context logic
        context_str = json.dumps(context, indent=2)
        system_prompt = system_prompt.replace("<<CONTEXT>>", context_str)
        
        try:
            # Call LLM
            user_msg = f"User Request: {prompt}\nJSON Plan:"
            
            # Check capabilities
            import inspect
            sig = inspect.signature(self.llm.generate_response)
            if 'system_prompt' in sig.parameters:
                response_text = await asyncio.to_thread(self.llm.generate_response, user_msg, system_prompt=system_prompt)
            else:
                response_text = await asyncio.to_thread(self.llm.generate_response, f"{system_prompt}\n\n{user_msg}")
            
            # Don't log raw output to prevent any leakage
            # logger.info(f"Raw Planner Output: {response_text}")

            # Robust JSON Extraction - extract ONLY the JSON part
            # 1. Strip Markdown
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # 2. Extract from first { to last }
            start_idx = clean_text.find('{')
            end_idx = clean_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                clean_text = clean_text[start_idx : end_idx + 1]
            
            # Parse
            plan = json.loads(clean_text)
            return plan

        except Exception as e:
            logger.error(f"Planning failed/Parsing failed: {e}")
            logger.error(f"Failed Text: {response_text if 'response_text' in locals() else 'None'}")
            # Return signal to trigger conversational response instead of returning JSON
            return {"steps": [], "_error": True}

    def _run_tool(self, tool_name: str, action: str, params: Dict):
        """Dynamic dispatch to tools."""
        tool_instance = getattr(self, tool_name.replace("_agent", "").replace("_controller", ""), None)
        
        # Map nice names to internal attributes
        if tool_name == "system_control": tool_instance = self.system
        elif tool_name == "file_manager": tool_instance = self.files
        elif tool_name == "browser_agent": tool_instance = self.browser
        elif tool_name == "input_controller": tool_instance = self.input

        if not tool_instance:
            raise ValueError(f"Unknown tool: {tool_name}")
            
        method = getattr(tool_instance, action, None)
        if not method:
            raise ValueError(f"Unknown action: {action} in {tool_name}")
            
        return method(**params)
