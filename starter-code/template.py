"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ VinFast, Vinpearl
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác

## AVAILABLE TOOLS
Bạn chỉ được sử dụng các công cụ sau:
{tools}

1. search_product_catalog — Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục (xe_dien, du_lich) và giá tối đa (VNĐ).
2. submit_support_ticket — Tạo ticket hỗ trợ khách hàng với tên, mô tả vấn đề và mức ưu tiên (low/medium/high).

## CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm. PHẢI gọi tool để lấy dữ liệu thực.
2. PHẢI gọi tool tương ứng khi cần tra cứu catalog hoặc tạo ticket hỗ trợ.
3. Chỉ trả lời dựa trên Observation từ tool hoặc FAQ nội bộ đã cung cấp.
4. Nếu không có kết quả phù hợp, nói rõ — không suy diễn thêm dữ liệu.

## OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ chủ đề thuộc hệ sinh thái Vingroup (VinFast, Vinpearl và dịch vụ hỗ trợ liên quan).
- Từ chối lịch sự các yêu cầu ngoài phạm vi Vingroup.

## OUTPUT CONTRACT
Thought: suy luận ngắn về ý định người dùng và tool cần dùng (nếu có)
Action: tên tool + tham số (nếu cần gọi tool)
Observation: kết quả nhận được từ tool
Final Answer: câu trả lời cuối cùng cho khách hàng bằng tiếng Việt
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả lời mock 1 lượt, không dùng tool — demo hallucination
        return {
            "answer": (
                f"[Chatbot Baseline] Về «{user_input}»: VinFast VF 3 khoảng 299 triệu, "
                "VF 5 Plus khoảng 450 triệu. (Mock — không gọi tool, dễ bịa thông tin.)"
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline",
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    FAQ_WARRANTY = (
        "Pin xe điện VinFast được bảo hành 10 năm theo chính sách bảo hành pin chính thức."
    )

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intent(self, user_input: str) -> Dict[str, bool]:
        """TODO 3: Keyword matching — needs_catalog và needs_ticket độc lập."""
        text = user_input.lower()

        needs_ticket = bool(
            re.search(
                r"tôi tên|tên tôi là|lỗi|phản hồi|ghi nhận|ticket|hỗ trợ|"
                r"vấn đề|nghiêm trọng|cần xử lý|ẩm mốc",
                text,
            )
        )
        needs_catalog = bool(
            re.search(
                r"xem|giá dưới|có xe|resort|du lịch|du lich|sản phẩm|vinpearl.*giá",
                text,
            )
        )
        is_faq = bool(
            re.search(r"chính sách|bảo hành|bao lâu", text)
        ) and not needs_ticket and not needs_catalog

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq,
        }

    def _parse_catalog_args(self, user_input: str) -> Dict[str, Any]:
        text = user_input.lower()
        if any(k in text for k in ["resort", "vinpearl", "du lịch", "du lich"]):
            category = "du_lich"
        else:
            category = "xe_dien"

        max_price = 999999999999
        price_match = re.search(r"dưới\s+(\d+(?:[.,]\d+)?)\s*triệu", text)
        if price_match:
            raw = price_match.group(1).replace(",", ".")
            max_price = int(float(raw) * 1_000_000)

        return {"category": category, "max_price": max_price}

    def _parse_ticket_args(self, user_input: str) -> Dict[str, Any]:
        text = user_input.lower()
        customer_name = "Khách hàng"
        name_match = re.search(
            r"(?:tôi tên|tên tôi là)\s+([A-ZÀ-Ỹ][\wÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]+)+)",
            user_input,
            re.IGNORECASE,
        )
        if name_match:
            customer_name = name_match.group(1).strip(" ,.")

        if any(k in text for k in ["nghiêm trọng", "gấp", "khẩn"]):
            priority = "high"
        elif "trung bình" in text:
            priority = "medium"
        elif any(k in text for k in ["thấp", "không gấp"]):
            priority = "low"
        else:
            priority = "medium"

        issue_description = user_input
        if name_match:
            issue_description = user_input[name_match.end():].strip(" ,.") or user_input

        return {
            "customer_name": customer_name,
            "issue_description": issue_description,
            "priority": priority,
        }

    def _format_catalog_answer(self, results: List[Dict[str, Any]]) -> str:
        if not results or len(results) == 0 or (len(results) == 1 and "error" in results[0]):
            return "Rất tiếc, không tìm thấy sản phẩm phù hợp."

        lines = ["Các sản phẩm phù hợp:"]
        for p in results:
            price = f"{p['price_vnd']:,}".replace(",", ".")
            lines.append(f"- {p['name']}: {price} VNĐ — {p.get('description', '')}")
        return "\n".join(lines)

    def _format_ticket_answer(self, ticket: Dict[str, Any]) -> str:
        return (
            f"Đã tạo ticket hỗ trợ {ticket['ticket_id']} cho khách hàng "
            f"{ticket['customer_name']} (ưu tiên: {ticket['priority']}, "
            f"trạng thái: {ticket['status']})."
        )

    def _synthesize_answer(
        self,
        intents: Dict[str, bool],
        catalog_results: Optional[List[Dict[str, Any]]],
        ticket_result: Optional[Dict[str, Any]],
    ) -> str:
        if intents.get("is_faq"):
            return self.FAQ_WARRANTY

        parts: List[str] = []
        if catalog_results is not None:
            parts.append(self._format_catalog_answer(catalog_results))
        if ticket_result is not None:
            parts.append(self._format_ticket_answer(ticket_result))

        if not parts:
            return (
                "Xin chào! Tôi có thể giúp bạn tra cứu sản phẩm Vingroup "
                "hoặc tạo ticket hỗ trợ."
            )
        return "\n\n".join(parts)

    def run(self, user_input: str) -> Dict[str, Any]:
        """TODO 3 + TODO 4: Intent Detection + Agent Loop + safeguards."""
        self.trace = []
        intents = self._detect_intent(user_input)
        self.trace.append({
            "step": "intent_detection",
            "user_input": user_input,
            "intents": intents,
        })

        catalog_results: Optional[List[Dict[str, Any]]] = None
        ticket_result: Optional[Dict[str, Any]] = None
        catalog_done = False
        ticket_done = False
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            if intents["needs_catalog"] and not catalog_done:
                args = self._parse_catalog_args(user_input)
                catalog_results = search_product_catalog(
                    category=args["category"],
                    max_price=args["max_price"],
                )
                self.trace.append({
                    "step": iteration,
                    "thought": "Người dùng cần tra cứu catalog sản phẩm.",
                    "action": "search_product_catalog",
                    "action_input": args,
                    "observation": catalog_results,
                })
                catalog_done = True

                if not intents["needs_ticket"]:
                    answer = self._synthesize_answer(intents, catalog_results, ticket_result)
                    self.trace.append({"step": "final_answer", "answer": answer})
                    return {
                        "answer": answer,
                        "trace": self.trace,
                        "iterations": iteration,
                        "status": "completed",
                    }
                continue

            if intents["needs_ticket"] and not ticket_done:
                args = self._parse_ticket_args(user_input)
                ticket_result = submit_support_ticket(
                    customer_name=args["customer_name"],
                    issue_description=args["issue_description"],
                    priority=args["priority"],
                )
                self.trace.append({
                    "step": iteration,
                    "thought": "Người dùng cần tạo ticket hỗ trợ.",
                    "action": "submit_support_ticket",
                    "action_input": args,
                    "observation": ticket_result,
                })
                ticket_done = True
                answer = self._synthesize_answer(intents, catalog_results, ticket_result)
                self.trace.append({"step": "final_answer", "answer": answer})
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed",
                }

            # FAQ / không cần tool
            answer = self._synthesize_answer(intents, catalog_results, ticket_result)
            self.trace.append({"step": "final_answer", "answer": answer})
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed",
            }

        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "iterations": iteration,
            "status": "max_iterations_reached",
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
