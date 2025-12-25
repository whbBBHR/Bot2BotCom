import anthropic
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Bot 1 (Claude) receives a message
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Bot 2 (OpenAI) responds
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chatbot_conversation(initial_prompt, turns=3):
    conversation_history = []
    current_message = initial_prompt
    
    for i in range(turns):
        # Claude responds
        claude_response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": current_message}]
        )
        claude_text = claude_response.content[0].text
        conversation_history.append({"speaker": "Claude", "message": claude_text})
        
        # GPT responds to Claude
        gpt_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": claude_text}]
        )
        gpt_text = gpt_response.choices[0].message.content
        conversation_history.append({"speaker": "GPT", "message": gpt_text})
        
        current_message = gpt_text
    
    return conversation_history

if __name__ == "__main__":
    print("=" * 80)
    print("Bot2Bot Conversation - Claude vs GPT-4o")
    print("=" * 80 + "\n")
    
    # Get user input
    initial_prompt = input("Enter your discussion question: ").strip()
    
    if not initial_prompt:
        print("Error: Please enter a valid question.")
        exit(1)
    
    try:
        turns_input = input("How many conversation turns? (default: 3): ").strip()
        turns = int(turns_input) if turns_input else 3
    except ValueError:
        print("Invalid input. Using default 3 turns.")
        turns = 3
    
    print(f"\nStarting {turns}-turn conversation...\n")
    
    conversation = chatbot_conversation(initial_prompt, turns=turns)
    
    print("=== Bot2Bot Conversation ===\n")
    for entry in conversation:
        print(f"{entry['speaker']}:")
        print(f"{entry['message']}\n")
        print("-" * 80 + "\n")