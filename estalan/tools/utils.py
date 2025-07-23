from langgraph.graph import StateGraph
from tenacity import retry, retry_if_result, stop_after_attempt

from estalan.logging_config import get_logger

logger = get_logger(__name__)


def add_graph_components(
    graph_builder: StateGraph, nodes, edges, edges_with_conditions
):
    try:
        for node_name, node_func in nodes:
            graph_builder.add_node(node_name, node_func)

        for edge_src, edge_dest in edges:
            graph_builder.add_edge(edge_src, edge_dest)

        for src, path, *path_map in edges_with_conditions:
            if path_map:
                graph_builder.add_conditional_edges(src, path, *path_map)
            else:
                graph_builder.add_conditional_edges(src, path)

            logger.info("Graph components added successfully")
        return graph_builder

    except Exception as e:
        logger.error(f"Error adding graph components: {str(e)}")
        raise


async def noop(*args, **kwargs):
    pass


def is_result_empty(result):
    return not result


def retry_on_api_empty():
    return retry(
        retry=retry_if_result(is_result_empty),  # 결과값이 비었을 경우에만 재시도.
        stop=stop_after_attempt(2),  # 최대 2번 실행.
        retry_error_callback=lambda future: None,  # 최종 실패 시, None 반환.
    )
