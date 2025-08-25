import langgraph_supervisor


def create_supervisor(*args, **kwargs):
    agent = langgraph_supervisor.create_supervisor(*args, **kwargs)
    return agent