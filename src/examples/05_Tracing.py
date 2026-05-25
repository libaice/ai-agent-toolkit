import anthropic
import json
import time
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

# 简单的 Trace 收集器
class Tracer:
    def __init__(self):
        self.steps = []
        self.start_time = time.time()
    
    def log(self, step_type: str, data: dict, duration_ms: float = 0):
        self.steps.append({
            "step": len(self.steps) + 1,
            "type": step_type,
            "data": data,
            "duration_ms": round(duration_ms, 1),
            "timestamp": datetime.now().isoformat()
        })
    
    def print_trace(self):
        print("\n" + "=" * 50)
        print("EXECUTION TRACE")
        print("=" * 50)
        total_time = (time.time() - self.start_time) * 1000
        
        for step in self.steps:
            print(f"\nStep {step['step']}: [{step['type']}] ({step['duration_ms']}ms)")
            if step['type'] == 'llm_call':
                print(f"  Input tokens: {step['data'].get('input_tokens')}")
                print(f"  Output tokens: {step['data'].get('output_tokens')}")
                print(f"  Stop reason: {step['data'].get('stop_reason')}")
            elif step['type'] == 'tool_call':
                print(f"  Tool: {step['data'].get('tool_name')}")
                print(f"  Input: {step['data'].get('input')}")
                print(f"  Success: {step['data'].get('success')}")
            elif step['type'] == 'guardrail':
                print(f"  Check: {step['data'].get('check')}")
                print(f"  Passed: {step['data'].get('passed')}")
        
        print(f"\nTotal steps: {len(self.steps)}")
        print(f"Total time: {total_time:.0f}ms")
        
        # 找最慢的步骤
        if self.steps:
            slowest = max(self.steps, key=lambda x: x['duration_ms'])
            print(f"Slowest step: Step {slowest['step']} ({slowest['type']}, {slowest['duration_ms']}ms)")

# 带 trace 的完整 agent 运行
def run_traced_agent(user_query: str):
    tracer = Tracer()
    messages = [{"role": "user", "content": user_query}]
    
    tools = [
        {
            "name": "get_spot_price",
            "description": "Get current price",
            "input_schema": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"]
            }
        }
    ]
    
    while True:
        t0 = time.time()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            tools=tools,
            messages=messages
        )
        duration = (time.time() - t0) * 1000
        
        tracer.log("llm_call", {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "stop_reason": response.stop_reason
        }, duration)
        
        if response.stop_reason == "end_turn":
            final = next((b.text for b in response.content if hasattr(b, 'text')), "")
            tracer.log("final_output", {"text": final[:100]})
            tracer.print_trace()
            return final
        
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    t0 = time.time()
                    try:
                        # mock execution
                        result = json.dumps({"price": 67450})
                        duration = (time.time() - t0) * 1000
                        tracer.log("tool_call", {
                            "tool_name": block.name,
                            "input": block.input,
                            "success": True
                        }, duration)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                    except Exception as e:
                        duration = (time.time() - t0) * 1000
                        tracer.log("tool_call", {
                            "tool_name": block.name,
                            "input": block.input,
                            "success": False,
                            "error": str(e)
                        }, duration)
            
            messages.append({"role": "user", "content": tool_results})

# 运行
run_traced_agent("BTC 现在多少钱，适合做市吗？")
