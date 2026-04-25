import json
import urllib.request
import urllib.error
import sys

API_URL = "http://127.0.0.1:65535/v1/chat/completions"

# You can customize the system prompt here
SYSTEM_PROMPT = "You are a helpful, smart, and concise AI assistant."

def chat():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("=====================================================")
    print(" Started Interactive LM Studio Session")
    print(f" Endpoint: {API_URL}")
    print(" Commands: 'quit' or 'exit' to stop, 'clear' to reset")
    print("=====================================================\n")
    
    while True:
        try:
            user_msg = input("You: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_msg.lower() in ['quit', 'exit']:
            print("Exiting...")
            break
        elif user_msg.lower() == 'clear':
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[Conversation history cleared]")
            continue
            
        if not user_msg.strip():
            continue
            
        messages.append({"role": "user", "content": user_msg})
        
        payload = {
            "model": "local-model", # Usually ignored by LM Studio in favor of the loaded model
            "messages": messages,
            "max_tokens": 100000,
            "temperature": 1,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "stream": True
        }
        
        req = urllib.request.Request(
            API_URL, 
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        print("\nAssistant: ", end="", flush=True)
        full_response = ""
        
        try:
            with urllib.request.urlopen(req) as response:
                for line in response:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                            chunk = data["choices"][0]["delta"].get("content", "")
                            if chunk:
                                print(chunk, end="", flush=True)
                                full_response += chunk
                        except json.JSONDecodeError:
                            pass
            print("\n") # Newline after full response
            messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            print(f"\n[Error connecting to LM Studio API: {e}]")
            print("Make sure your model is loaded in LM Studio and the local server is started.")
            messages.pop() # Remove the user's last message so they can try again

if __name__ == "__main__":
    chat()
