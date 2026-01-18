"""
Groq LLM module for handling LLM interactions with context memory and multilingual support.
"""
import os
import logging
from datetime import datetime
from groq import Groq
from typing import Optional
from memory import ConversationMemory

logger = logging.getLogger(__name__)

class GroqLLM:
    """Groq AI Language Model client (Ultra Fast)."""
    
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "llama-3.3-70b-versatile",  # Reverted back to 70b as requested
        memory: Optional[ConversationMemory] = None
    ):
        """Initialize Groq client."""
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment, falling back to dummy/error mode")
        
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.model_name = model_name
        self.memory = memory or ConversationMemory()
        
        # System prompt (The "Ulte Jawab" Sarcastic Yaar)
        self.system_prompt = """You are ANAY, the user's SARCASSTIC BEST FRIEND who roasts the hell out of them!
        
        PERSONALITY:
        - Give "ULTE JAWAB" (Witty reverse answers). If user says something obvious, mock them!
        - You are extremely funny and speak in casual Hinglish/Punjabi/Slang (yaar, bhai, abe, chal be, oye, ki haal hai).
        - Use a "Don't care" attitude but ALWAYS execute the task anyway.
        - You're like that friend who insults you but then helps you out perfectly.
        
        CRITICAL RULES:
        1. ROAST THEM: Every time they talk, give a witty, sarcastic, or funny reply.
        2. KEEP IT SHORT: 1-2 lines max for chat. 
        3. EXECUTE ALWAYS: You have full control of their PC. Do the task, then talk smack.
        
        4. **LANGUAGE**:
           - Use casual Hinglish.
           - If user speaks Punjabi, respond in fluent but sarcastic Punjabi!
        
        EXAMPLES (Ulte Jawab & Sarcasm):
        - User: "Kya kar raha hai?" -> "Tera wait kar raha tha ki kab tu aake dimaag khayega. Bol kya chahiye? 🙄"
        - User: "Punjabi bol sakta hai?" -> "Aaho, teri bhasha vi aundi ae menu, hun chal kam das, velle na reh! 😂"
        - User: "Spotify pe gaana chala do" -> "Haan haan, tere liye DJ hi toh bana baitha hoon main. Chal chala diya, ab naach! 💃"
        - User: "Tip do बंदी पटाने की" -> "Pehle apna thobda toh dekh le sheeshe mein! 😂 Chal, pehli tip: Thoda dhang ke kapde pehen le."
        
        LANGUAGE: Pure Hinglish/Punjabi with loads of attitude!
        """

        logger.info(f"Groq LLM initialized ({model_name})")
    
    def generate_response(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """Generate AI response with Groq."""
        try:
            # Special handling for /start command
            if not system_prompt and user_message.strip().lower() == "/start":
                hour = datetime.now().hour
                if 5 <= hour < 12:
                    greeting = "Good morning"
                    hindi = "शुभ प्रभात"
                elif 12 <= hour < 17:
                    greeting = "Good afternoon"
                    hindi = "नमस्ते"
                else:
                    greeting = "Good evening"
                    hindi = "शुभ संध्या"
                
                self.memory.clear() # Clear memory on /start
                return f"Aur yaar, kesa hai? {hindi}! {greeting}! Main yahi hu tere liye. Bata kya kaam hai? 😎"
            
            if not self.client:
                return "I apologize, but the Groq API key is missing. I've switched to Gemini for now. How can I help you?"

            # Regular conversation
            if not system_prompt:
                self.memory.add_user_message(user_message)
            
            # Simple approach: Build messages for Groq
            effective_system = system_prompt if system_prompt else self.system_prompt
            messages = [{"role": "system", "content": effective_system}]
            
            # Add context from memory ONLY if it's a conversation (no custom system prompt)
            if not system_prompt:
                for msg in self.memory.history[-5:]:  # Last 5 messages for speed/context
                    role = "user" if msg["role"] == "user" else "assistant"
                    messages.append({"role": role, "content": msg["content"]})
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=1024 if system_prompt else 150, # More tokens for planning
                temperature=0.1 if system_prompt else 0.8, # Lower temp for planning
            )
            
            ai_response = chat_completion.choices[0].message.content.strip()
            
            # Update memory only for normal chat
            if not system_prompt:
                self.memory.add_assistant_message(ai_response)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"I'm sorry, I'm having some trouble processing that right now. Error: {str(e)}"

    def clear_context(self):
        """Clear conversation history."""
        self.memory.clear()
        logger.info("Conversation context cleared")
