class UnionFind:
    def __init__(self, n):
        self.par, self.rank = [i for i in range(n)], [1] * n
    def find_parent(self, v1):
        while v1 != self.par[v1]:
            self.par[v1] = self.par[self.par[v1]]
            v1 = self.par[v1]
        return v1
    def union(self, v1, v2):
        p1, p2 = self.find_parent(v1), self.find_parent(v2)
        if p1 == p2: return False
        if self.rank[p1] > self.rank[p2]:
            self.par[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.par[p1] = p2
            self.rank[p2] += self.rank[p1]
        return True

    # # for maxNumEdgesToRemove
    # def __init__(self, nodes: int):
    #     self.parents, self.ranks = [i for i in range(nodes)], [1 for _ in range(nodes)]
    # def find(self, node: int):
    #     node_parents = []
    #     while node != self.parents[node]:
    #         node_parents.append(node)
    #         node = self.parents[node]
    #     for parent_node in node_parents: self.parents[parent_node] = node
    #     return node
    #
    # def union_cycle(self, node_1: int, node_2: int):
    #     node_1_parent, node_2_parent = self.find(node_1), self.find(node_2)
    #     if node_1_parent != node_2_parent:
    #         if self.ranks[node_1_parent] > self.ranks[node_2_parent]:
    #             self.ranks[node_1_parent] += self.ranks[node_2_parent]
    #             self.parents[node_2_parent] = node_1_parent
    #         else:
    #             self.ranks[node_2_parent] += self.ranks[node_1_parent]
    #             self.parents[node_1_parent] = node_2_parent
    #         return False
    #     else:
    #         return True