from langgraph.graph import StateGraph


class AlanAgentBuilder(object):
    def __init__(self, state_schema):
        self._state_graph = StateGraph(state_schema)

    @property
    def state_graph(self):
        """내부 StateGraph 인스턴스에 접근할 수 있는 프로퍼티"""
        return self._state_graph

    def add_edge(self, from_node, to_node):
        """
        StateGraph에 엣지를 추가합니다.
        
        Args:
            from_node (str): 시작 노드의 이름
            to_node (str): 도착 노드의 이름
            
        Returns:
            self: 메서드 체이닝을 위한 self 반환
        """
        self._state_graph.add_edge(from_node, to_node)
        return self

    def add_node(self, name, node_func):
        """
        StateGraph에 노드를 추가합니다.
        
        Args:
            name (str): 노드의 이름
            node_func (callable): 노드에서 실행할 함수
            
        Returns:
            self: 메서드 체이닝을 위한 self 반환
        """
        self._state_graph.add_node(name, node_func)
        return self

    def compile(self, **kwargs):
        """
        StateGraph를 컴파일합니다.
        
        Args:
            **kwargs: compile 메서드에 전달할 추가 인자들
            
        Returns:
            컴파일된 그래프
        """
        return self._state_graph.compile(**kwargs)
