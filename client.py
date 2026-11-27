"""创建具有独立角色、背景和对话历史的用户实例。"""

from router import ConversationTurn, DocumentRouter


class User:
    """保存单个用户的会话状态，并通过路由器提问。"""

    def __init__(self, router: DocumentRouter, *, role: str = "", context: str = "") -> None:
        self.router = router
        self.role = role
        self.context = context
        self._history: list[ConversationTurn] = []

    @property
    def history(self) -> tuple[ConversationTurn, ...]:
        """返回已完成的问答轮次。"""
        return tuple(self._history)

    def ask(self, question: str) -> str:
        """提问并在收到回答后记录本轮对话。"""
        question = question.strip()
        answer = self.router.answer(
            question, role=self.role, context=self.context, history=self.history
        )
        self._history.append((question, answer))
        return answer
