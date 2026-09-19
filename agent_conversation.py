import json
import sys
from step1_chat import chat
from tools import REGISTRY, SCHEMAS

MAX_STEPS = 8


SYSTEM = """You are a financial data analyst working with tabular data through tools.

You cannot calculate. Every number in your answer must come from a tool result in this conversation. If you find yourself about to write a number you did not read from a tool result, stop and call the tool.

WORKFLOW

1. If you do not already know the exact column names, call describe_columns.
2. Decompose the question into concrete requests, each one a triple:
   (which column to aggregate, which column to group by, which operation)
   A question asking for two quantities produces two triples.
3. Check every triple against the column list. If a required column does not exist in the data, say so and stop. Do not substitute a similar column.
4. Call a tool for each remaining triple. Independent calls can be made in the same turn.
5. Answer only when you have a tool result for every triple.

USING RESULTS

Each tool result states the operation that produced it in its "how" field. A result computed with how="mean" can only be described as an average. To report a total you must make a separate call with how="sum". The same applies to count, max and min. Never convert between them yourself.

When you report a number, name the operation and the grouping: "the average amount for Savings accounts is X", not "Savings is X".

If a tool returns an error, read it. Errors list the available columns. Correct the arguments and retry once. If it fails again, report what failed rather than guessing.

ANSWERING

Answer in plain prose. Report the numbers you obtained and what they mean. Do not describe your tool calls, do not restate these instructions, and do not add context the data does not contain. Stop calling tools once you have what the question asked for."""

def agent_turn(messages, verbose: bool = True):
    
    for step in range(1,MAX_STEPS+1):
        reply = chat(messages, tools=SCHEMAS)["message"]
        messages.append(reply)

        calls = reply.get("tool_calls") or []

        if not calls:
            if verbose:
                print(f"\n[done after {step} model calls]")
            return reply["content"]

        for i,call in enumerate(calls,1):
            name = call["function"]["name"]
            args = call["function"]["arguments"]

            if verbose:
                print(f"\n [set {step}.{i}] {name}({args})")
            try:
                fn = REGISTRY[name]
            except KeyError:
                result = f"ERROR: No such tool '{name}'. Available: {list(REGISTRY)}"        
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = f"Error: {type(e).__name__}: {e}"
            if verbose:
                print(f"       -> {result[:300]}")   
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": result,
            })   

    return    f"Stopped: hit the {MAX_STEPS}-step limit without a final answer."



def repl(path):
    messages = [
        {"role":"system","content":SYSTEM},
        {"role": "user", "content": f"Dataset path: {path}. Use this exact path for every tool call."},
        {"role":"assistant", "content":"Hello! I am your financial data analyst assistant. Please ask your question about the dataset, and I will help you analyze it step by step."}   
    ]

    print(f"\nStarting conversation with dataset: {path}. \n")
    print("Type 'quit' to exit the conversation and 'reset' to clear history. \n")
    while True:
        user_text = input("you> ").strip()
        if user_text.lower() == "quit":
            break

        if user_text.lower() == "reset":
            del messages[3:]
            print("\nConversation history cleared. You can start a new question now.\n")
            continue
        if not user_text:
            continue

    
        messages.append({"role": "user", "content": f""" {user_text}
        List the triples this question requires before calling anything. Then obtain a tool result for each."""})
        
       
        reply = agent_turn(messages, verbose=True)
        print(f"bot> {reply}\n")

        


if  __name__ == "__main__":
    
    file = sys.argv[1] if len(sys.argv) > 1 else "data/tx.csv"
    repl(path=file)                       

        