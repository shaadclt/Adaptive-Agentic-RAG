from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

def format_response(result):
    """
    Format the response from the graph for better readability
    """
    if isinstance(result,dict) and "generation" in result:
        return result["generation"]
    elif isinstance(result,dict) and "answer" in result:
        return result["answer"]
    else:
        return str(result)
    
    
def main():
    print("=" * 60)
    print("🤖 Agentic Adaptive RAG")
    print("=" * 60)
    print("Welcome! Ask me anything or type 'quit', 'exit' or 'bye' to stop.")
    print("-" * 60)
    
    while True:
        try:
            # Get User Input
            user_question = input("\n💬 You: ").strip()
            
            # Check for exit commands
            if user_question.lower() in ["quit", "exit", "bye", "q"]:
                print("\n👋 Goodbye! Thanks for chatting!")
                break
            
            # Skip empty inputs
            if not user_question:
                print("Please enter a question.")
                continue
            
            # Show processing indicator
            print("\n🤔 Bot: Thinking...")
            
            # Process the question through the graph
            result = app.invoke(input={"question":user_question})
            
            # Format and display the response
            response = format_response(result)
            print(f"🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"\n❌ Sorry, I encountered an error: {str(e)}")
            print("Please try asking your question again.")
            
if __name__ == "__main__":
    main()
    
"""
The main application file provides a simple entry point for testing and running our Agentic Adaptive RAG system. 
It loads environment variables, imports our compiled graph application, and demonstrates the working of the Advanced RAG Chatbot through the terminal.
"""