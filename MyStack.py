import collections


class MyStack:

    """
    225. Implement Stack using Queues
    Implement a last-in-first-out (LIFO) stack using only two queues.
    The implemented stack should support all the functions of a normal stack (push, top, pop, and empty).

    Implement the MyStack class:

    void push(int x) Pushes element x to the top of the stack.
    int pop() Removes the element on the top of the stack and returns it.
    int top() Returns the element on the top of the stack.
    boolean empty() Returns true if the stack is empty, false otherwise.
    Notes:

    You must use only standard operations of a queue, which means that only push to back, peek/pop from front, size and is empty operations are valid.
    Depending on your language, the queue may not be supported natively. You may simulate a queue using a list or deque (double-ended queue) as long as you use only a queue's standard operations.

    Example 1:
    Input
    ["MyStack", "push", "push", "top", "pop", "empty"]
    [[], [1], [2], [], [], []]
    Output
    [null, null, null, 2, 2, false]

    Explanation
    MyStack myStack = new MyStack();
    myStack.push(1);
    myStack.push(2);
    myStack.top(); // return 2
    myStack.pop(); // return 2
    myStack.empty(); // return False
    """
    """
    Hints: use a deque, fix push after you append
    """
    """
    def __init__(self):
        self.q = collections.deque()

    def pushMyStack(self, x: int) -> None:
        self.q.append(x)
        for _ in range(len(self.q) -1):
            self.q.append(self.q.popleft())

    def popMyStack(self) -> int:
        return self.q.popleft()

    def topMyStack(self) -> int:
        return self.q[0]

    def emptyMyStack(self) -> bool:
        return len(self.q) == 0
    """

    def __int__(self):
        None
    def pushMyStack(self, x: int) -> None:
        None
    def popMyStack(self) -> int:
        return 0
    def topMyStack(self) -> int:
        return 0
    def emptyMyStack(self) -> bool:
        return False
