"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
from pathlib import Path

# Load .env file (tìm từ thư mục starter-code lên một cấp)
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""


class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> str:
        """
        Trả về câu trả lời từ LLM 1 lượt, không sử dụng tool.
        Nếu có API key thì gọi Gemini, không thì trả về câu trả lời tĩnh.
        """
        if self.api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                prompt = (
                    "Bạn là một chatbot tư vấn du lịch. Hãy trả lời câu hỏi sau "
                    f"mà KHÔNG sử dụng tools hay internet: {user_input}"
                )
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                return f"[Lỗi khi gọi Gemini API: {e}]"
        else:
            # Fallback: câu trả lời tĩnh khi không có API key
            return (
                "Xin lỗi, tôi không có đủ thông tin thực tế để trả lời câu hỏi này. "
                "Vui lòng kiểm tra lại với đại lý du lịch hoặc website chính thức."
            )


class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    def __init__(self, max_iterations: int = 5, api_key=None):
        self.max_iterations = max_iterations
        self.trace = []
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def _call_llm(self, messages: list, retries: int = 3) -> str:
        """Gọi Gemini với danh sách messages. Tự retry khi gặp rate-limit (429)."""
        import time
        from google import genai
        client = genai.Client(api_key=self.api_key)
        full_prompt = "\n".join(messages)
        for attempt in range(1, retries + 1):
            try:
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=full_prompt,
                )
                return response.text
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 60 * attempt  # 60s, 120s, 180s
                    print(f"  [Rate limit] Chờ {wait}s rồi thử lại (lần {attempt}/{retries})...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Vượt quá số lần retry do rate limit.")

    def run(self, user_input: str) -> str:
        """
        Chạy vòng lặp ReAct:
        1. Khởi tạo lịch sử conversation / trace
        2. Lặp tối đa max_iterations lần
        3. Phân tích Thought / Action từ phản hồi Agent
        4. Thực thi Tool trong TOOL_MAP nếu có Action
        5. Ghi lại Observation và lặp lại cho tới khi có Final Answer
        """
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []

        # Chuẩn bị system prompt với danh sách tool kèm tham số
        def _fmt_tool(t):
            params = ", ".join(
                f"{k}: {v}" for k, v in t.get("parameters", {}).items()
            )
            return f"  - {t['name']}({params}): {t['description']}"

        tools_desc = "\n".join([_fmt_tool(t) for t in TOOL_DEFINITIONS])
        system_prompt = SYSTEM_PROMPT.format(tools=tools_desc)

        # Khởi tạo lịch sử messages
        messages = [system_prompt, f"User: {user_input}"]

        # TODO 2: Thiết lập vòng lặp while iteration < self.max_iterations
        for iteration in range(1, self.max_iterations + 1):
            # Gọi LLM
            try:
                response_text = self._call_llm(messages)
            except Exception as e:
                error_msg = f"[Lỗi khi gọi LLM ở iteration {iteration}: {e}]"
                self.trace.append({"iteration": iteration, "error": str(e)})
                return error_msg

            # TODO 3: Phân tích Thought / Action từ Agent
            thought_match = re.search(r"Thought:\s*(.*)", response_text)
            thought = thought_match.group(1).strip() if thought_match else ""

            # Ghi trace phản hồi thô
            step_log = {
                "iteration": iteration,
                "thought": thought,
                "raw_response": response_text,
            }

            # Kiểm tra Final Answer
            final_match = re.search(r"Final Answer:\s*(.*)", response_text, re.DOTALL)
            if final_match:
                final_answer = final_match.group(1).strip()
                step_log["final_answer"] = final_answer
                self.trace.append(step_log)
                return final_answer

            # TODO 4: Thực thi Tool trong TOOL_MAP nếu có Action
            # Dùng balanced-braces extraction để xử lý JSON lồng nhau
            observation = "Không có action nào được phát hiện."
            action_json = None

            action_start = re.search(r"Action:\s*(\{)", response_text)
            if action_start:
                start_idx = action_start.start(1)
                depth = 0
                end_idx = None
                for i, ch in enumerate(response_text[start_idx:], start=start_idx):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break
                if end_idx:
                    raw_json = response_text[start_idx:end_idx]
                    # Chuẩn hóa dấu nháy đặc biệt thành ASCII
                    raw_json = raw_json.replace("\u201c", '"').replace("\u201d", '"')
                    try:
                        action_json = json.loads(raw_json)
                        tool_name = action_json.get("name")
                        args = action_json.get("args", {})
                        tool_func = TOOL_MAP.get(tool_name)
                        if tool_func:
                            result = tool_func(**args)
                            observation = json.dumps(result, ensure_ascii=False, indent=2)
                        else:
                            observation = f"Tool '{tool_name}' không tồn tại trong TOOL_MAP."
                    except json.JSONDecodeError as e:
                        observation = f"Không thể parse Action JSON: {e} | raw: {raw_json[:200]}"
                    except TypeError as e:
                        observation = f"Lỗi tham số khi gọi tool: {e}"
                    except Exception as e:
                        observation = f"Lỗi khi thực thi tool: {e}"

            # TODO 5: Ghi lại Observation và lặp lại
            step_log["action"] = action_json
            step_log["observation"] = observation
            self.trace.append(step_log)

            # Cập nhật lịch sử messages cho iteration tiếp theo
            messages.append(response_text)
            messages.append(f"Observation: {observation}")

        # Vượt quá max_iterations
        self.trace.append({"status": "max_iterations_exceeded"})
        return "Đã vượt quá số vòng lặp tối đa mà không có Final Answer."


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()