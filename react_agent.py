"""
ReAct-style agent that works with text-based tool calling (Llama/open models).
This agent parses tool calls from model text output instead of using function calling API.
"""

import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a parsed tool call from model output."""
    name: str
    parameters: Dict[str, Any]
    raw_text: str


class ReActAgent:
    """
    ReAct-style agent that works with models outputting text-based tool calls.
    Supports Llama-style <|python_tag|> format and plain JSON tool calls.
    """
    
    def __init__(self, llm, tools: Dict[str, Any], max_iterations: int = 5):
        """
        Args:
            llm: The language model to use (ChatOpenAI or similar)
            tools: Dictionary of tool name -> tool function
            max_iterations: Maximum number of reasoning iterations
        """
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        
    def _create_react_prompt(self, query: str, tool_descriptions: str, history: List[str]) -> str:
        """Create a ReAct-style prompt for the model."""
        
        history_text = "\n".join(history) if history else ""
        
        prompt = f"""You are a helpful AI assistant that answers questions using available tools.

Available Tools:
{tool_descriptions}

Instructions:
1. Think step-by-step about how to answer the question
2. Use tools by outputting ONLY: {{"name": "tool_name", "parameters": {{"param": "value"}}}}
3. IMPORTANT: Match the exact parameter names shown in the tool descriptions
4. After using a tool, you'll receive the results
5. Once you have enough information, provide your final answer starting with "Final Answer:"

Question: {query}

{history_text}

Let's think step by step:"""
        
        return prompt
    
    def _parse_tool_calls(self, text: str) -> List[ToolCall]:
        """
        Parse tool calls from model output.
        Supports multiple formats:
        - <|python_tag|>{"name": "tool", "parameters": {...}}
        - {"name": "tool", "parameters": {...}}
        """
        tool_calls = []
        
        # Pattern 1: <|python_tag|> format
        python_tag_pattern = r'<\|python_tag\|>(\{[^}]+\})'
        matches = re.finditer(python_tag_pattern, text)
        
        for match in matches:
            try:
                json_str = match.group(1)
                data = json.loads(json_str)
                if "name" in data and "parameters" in data:
                    tool_calls.append(ToolCall(
                        name=data["name"],
                        parameters=data["parameters"],
                        raw_text=match.group(0)
                    ))
            except json.JSONDecodeError:
                continue
        
        # Pattern 2: Plain JSON without tags (fallback)
        if not tool_calls:
            json_pattern = r'\{["\']name["\']\s*:\s*["\']([^"\']+)["\']\s*,\s*["\']parameters["\']\s*:\s*\{[^}]+\}\}'
            matches = re.finditer(json_pattern, text)
            
            for match in matches:
                try:
                    data = json.loads(match.group(0))
                    if "name" in data and "parameters" in data:
                        tool_calls.append(ToolCall(
                            name=data["name"],
                            parameters=data["parameters"],
                            raw_text=match.group(0)
                        ))
                except json.JSONDecodeError:
                    continue
        
        return tool_calls
    
    def _check_for_final_answer(self, text: str) -> Optional[str]:
        """Check if the model has provided a final answer."""
        # Look for "Final Answer:" marker
        final_answer_pattern = r'Final Answer:\s*(.+?)(?:\n\n|$)'
        match = re.search(final_answer_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # If no tool calls and response looks like an answer, treat it as final
        if "Final Answer" not in text and len(text) > 50:
            tool_calls = self._parse_tool_calls(text)
            if not tool_calls and not text.startswith("<|python_tag|>"):
                return text.strip()
        
        return None
    
    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool and return its result."""
        if tool_call.name not in self.tools:
            return f"Error: Tool '{tool_call.name}' not found. Available tools: {list(self.tools.keys())}"
        
        try:
            tool_func = self.tools[tool_call.name]
            result = tool_func(**tool_call.parameters)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_call.name}: {str(e)}"
    
    def run(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the ReAct agent loop.
        
        Returns:
            Dict with 'answer', 'iterations', and 'history'
        """
        # Create tool descriptions
        tool_descriptions = "\n".join([
            f"- {name}: {func.__doc__ or 'No description'}"
            for name, func in self.tools.items()
        ])
        
        history = []
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print('='*60)
            
            # Create prompt with history
            prompt = self._create_react_prompt(query, tool_descriptions, history)
            
            # Get model response
            if verbose:
                print("\n🤔 Thinking...")
            
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            if verbose:
                print(f"\n📝 Model response:\n{response_text[:500]}...")
            
            # Check for final answer
            final_answer = self._check_for_final_answer(response_text)
            if final_answer:
                if verbose:
                    print(f"\n✅ Final answer found!")
                return {
                    'answer': final_answer,
                    'iterations': iteration + 1,
                    'history': history
                }
            
            # Parse and execute tool calls
            tool_calls = self._parse_tool_calls(response_text)
            
            if not tool_calls:
                # No tools and no final answer - treat response as final answer
                if verbose:
                    print("\n⚠️ No tool calls found, treating as final answer")
                return {
                    'answer': response_text,
                    'iterations': iteration + 1,
                    'history': history
                }
            
            # Execute tools
            for tool_call in tool_calls:
                if verbose:
                    print(f"\n🔧 Executing tool: {tool_call.name}")
                    print(f"   Parameters: {tool_call.parameters}")
                
                result = self._execute_tool(tool_call)
                
                if verbose:
                    print(f"\n📊 Tool result:\n{result[:300]}...")
                
                # Add to history
                history.append(f"Thought: {response_text}")
                history.append(f"Tool: {tool_call.name}({json.dumps(tool_call.parameters)})")
                history.append(f"Result: {result}")
        
        # Max iterations reached
        if verbose:
            print(f"\n⚠️ Reached maximum iterations ({self.max_iterations})")
        
        return {
            'answer': "I apologize, but I reached the maximum number of reasoning steps without finding a complete answer. Please try rephrasing your question.",
            'iterations': self.max_iterations,
            'history': history
        }
