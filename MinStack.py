import collections

class MinStack:

    """
    155. Min Stack
    Design a stack that supports push, pop, top,
    and retrieving the minimum element in constant time.
    Implement the MinStack class:
    MinStack() initializes the stack object.
    void push(int val) pushes the element val onto the stack.
    void pop() removes the element on the top of the stack.
    int top() gets the top element of the stack.
    int getMin() retrieves the minimum element in the stack.
    You must implement a solution with O(1) time complexity for each function.
    Ex1: Input ["MinStack","push","push","push","getMin","pop","top","getMin"]
    [[],[-2],[0],[-3],[],[],[],[]], Output: [null,null,null,null,-3,null,0,-2]
    Explanation
    MinStack minStack = new MinStack();
    minStack.push(-2);
    minStack.push(0);
    minStack.push(-3);
    minStack.getMin(); // return -3
    minStack.pop();
    minStack.top();    // return 0
    minStack.getMin(); // return -2
    """

    """
    def __init__(self):
        self.stack, self.minStack = [], []

    def pushMinStack(self, val: int) -> None:
        self.stack.append(val)
        val = min(val, self.minStack[-1] if self.minStack else val)
        self.minStack.append(val)

    def popMinStack(self) -> None:
        self.stack.pop()
        self.minStack.pop()

    def topMinStack(self) -> int:
        return self.stack[-1]

    def getMinStack(self) -> int:
        return self.minStack[-1]
    """
    """
    Hints: two lists, push min val to minstack, 
    """
    def __init__(self):
        return None
    def pushMinStack(self, val: int) -> None:
        return None
    def popMinStack(self) -> None:
        return None
    def topMinStack(self) -> int:
        return 0
    def getMinStack(self) -> int:
        return 0

# if __name__ == '__main__':
#     stack = MinStack()
#     stack.pushMinStack(-2)
#     stack.pushMinStack(0)
#     stack.pushMinStack(-3)
#
#     stack1 = stack.stack
#     minStack = stack.minStack
#     print(stack1)
#     print(minStack)
#     min = stack.getMinStack()
#     stack.popMinStack()
#     # stack.popMinStack()
#     stack.pushMinStack(1)
#     print(stack1)
#     print(minStack)